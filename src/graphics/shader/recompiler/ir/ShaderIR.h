#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
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
#include <cstdlib>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Emulated 64-bit FLAT apertures. Each contains 4 GiB of byte offsets and is
// outside guest global VA space. Keep queries and FLAT routing consistent: the values are
// upstream's Decoder::SharedApertureHigh / PrivateApertureHigh (ShaderDecoder.h).
inline constexpr uint32_t SharedApertureHigh = 0x80000000u;
inline constexpr uint32_t PrivateApertureHigh = 0x70000000u;

enum class ResourceKind {
	None,
	ScalarBuffer,
	ScalarAddress,
	Buffer,
	IndirectBuffer,
	Flat,
	FlatLocal,
	Global,
	Scratch,
	Lds,
	Gds,
	Image,
	Sampler
};

[[nodiscard]] constexpr bool IsAddressResourceKind(ResourceKind kind) {
	return kind == ResourceKind::ScalarAddress || kind == ResourceKind::Flat ||
	       kind == ResourceKind::FlatLocal || kind == ResourceKind::Global ||
	       kind == ResourceKind::Scratch;
}

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
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
	bool                    address_is_full                                       = false;
	bool                    data_signed                                           = false;
	bool                    typed                                                 = false;
	bool                    formatted                                             = false;
	bool                    image_has_mip                                         = false;
	bool                    image_r128                                            = false;
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		return !typed && data_bits == 32u &&
		       (formatted ? opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4
		                  : opcode == ValueOpcode::ReadConstBuffer ||
		                        opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4);
	}

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

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

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
	// From ResourceSpecialization::Buffer::write_tracked.
	bool                   write_tracked      = false;
	uint32_t               indirect_root              = NoIndirectBuffer;
	uint32_t               indirect_mapping_offset    = 0;
	uint32_t               indirect_search_iterations = 0;
	std::vector<uint32_t>  indirect_resources;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;

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
	bool                          atomic64          = false;
	bool                          depth_compare     = false;
	bool                          cube              = false;
	bool                          r128              = false;
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	// Sampled through the bindless image arrays (descriptor set 1) at the slot the translation
	// table gives for the handle's key; the mapping offset locates the heap's region and size.
	bool                          bindless          = false;
	// With bindless: the handle holds the eight T# dwords the shader loaded at run time (from a
	// material record, say). The shader looks the T# up in the runtime image table (set 1) and
	// queries its size from the dwords themselves; one resource serves every such T#.
	bool                          runtime = false;
	// A storage image of a 16-bit float format: its stores clamp finite values beyond the format's
	// range to the largest half (65504), as the guest GPU's conversion does (FloatStoreClamp).
	bool                          clamp_half_float = false;
	std::vector<uint32_t>         indirect_resources;

	bool operator==(const ImageResource& other) const = default;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct SamplerResource {
	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	// Native filtering/border variants share the original sampler's runtime descriptor.
	uint32_t snapshot_index        = 0;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;
	// Selected per draw by a GPU-computed key from a guest sampler heap: the shader indexes the
	// bindless sampler array with the region base and entry count at bindless_mapping_offset.
	bool     bindless                = false;
	uint32_t bindless_mapping_offset = 0;

	bool operator==(const SamplerResource& other) const = default;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

enum class TessellationAttribute {
	LocalOutput,
	ControlInput,
	ControlOutput,
	EvaluationInput,
	PatchOutput,
	Factor
};

enum class StageInputKind {
	VertexIndex,
	InvocationId,
	PrimitiveId,
	TessCoord,
	InstanceIndex,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	NumWorkgroups,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
	DispatchThreadCount,
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

// The specialization constant a pre-rasterization stage clamps its exported render target
// array index with: the bound layer count (0: no clamp).
inline constexpr uint32_t LayerLimitSpecId = 0x4c00u;

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

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct StageInput {
	StageInputKind kind            = StageInputKind::VertexIndex;
	uint32_t       location        = 0;
	uint32_t       component_count = 1;
	std::string    debug_name;
	bool           per_vertex = false;

	bool operator==(const StageInput& other) const = default;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
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
inline constexpr uint32_t ImageBindingCount           = 48u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	SharedMemory,
	Count,
};

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 49u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 56u);

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 7;
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
	const uint32_t group = stage == ShaderType::Pixel                    ? 1u
	                       : stage == ShaderType::TessellationControl    ? 2u
	                       : stage == ShaderType::TessellationEvaluation ? 3u
	                                                                     : 0u;
	return static_cast<uint32_t>(kind) +
	       group * static_cast<uint32_t>(DescriptorBindingKind::Count);
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

// KYTY_FASTDRAW_IMAGESET=1 (gt7-fastdraw-1112, off by default): the images and samplers of a
// shader's own descriptors are declared in descriptor set ImageSetDescriptorSet instead of set 0
// (set 1 is the bindless table, BindlessBindings.h), and the pipeline layouts follow
// (shaders.cpp). The switch is named here, in the recompiler's sources, so the shader disk cache's
// session key records its value (shader_cache_version.cmake): translations of either mode are
// never mixed. The driver's pipeline cache keys pipelines by their SPIR-V and layout already.
inline constexpr uint32_t ImageSetDescriptorSet = 2;
[[nodiscard]] inline bool FastDrawImageSet() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FASTDRAW_IMAGESET");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
// A binding that goes to set ImageSetDescriptorSet in that mode: images (sampled or storage) and
// samplers.
[[nodiscard]] constexpr bool ImageSetBindingKind(DescriptorBindingKind kind) {
	return kind == DescriptorBindingKind::Samplers ||
	       ImageBindingResourceClass(kind) != ImageResourceClass::None;
}
// The descriptor set of a shader's own binding of `kind`.
[[nodiscard]] inline uint32_t NativeDescriptorSet(DescriptorBindingKind kind) {
	return FastDrawImageSet() && ImageSetBindingKind(kind) ? ImageSetDescriptorSet : 0u;
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
			base = AtomicUintBinding + (image.atomic64 ? 5u : 0u);
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

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct DescriptorBinding {
	DescriptorBindingKind kind = DescriptorBindingKind::Buffers;
	std::vector<uint32_t> resources;

	bool operator==(const DescriptorBinding& other) const = default;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct BindingLayout {
	uint32_t                       push_data_start_dword = PushData::NoStart;
	uint32_t                       dispatch_thread_dword = PushData::NoStart;
	// Round 118: the partition target's two words (GetPartitionTarget), when the shader reads it.
	uint32_t                       partition_target_dword = PushData::NoStart;
	uint32_t                       memory_offset_dword = 0;
	uint32_t                       memory_offset_count = 0;
	// Some memory binding is write-tracked: WriteTrackDword() holds the bitmap's device address
	// (2 dwords), then the guest address of each memory binding's range start (2 dwords each; 0 =
	// not tracked in this draw).
	bool                           write_tracking      = false;
	std::vector<uint32_t>          user_data_registers;
	std::vector<DescriptorBinding> descriptors;

	[[nodiscard]] uint32_t BufferLengthDword() const {
		return memory_offset_dword + (memory_offset_count + 3u) / 4u;
	}
	[[nodiscard]] uint32_t WriteTrackDword() const {
		return BufferLengthDword() + memory_offset_count;
	}
	[[nodiscard]] uint32_t ShaderDataDwords() const {
		return WriteTrackDword() + (write_tracking ? 2u + 2u * memory_offset_count : 0u);
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

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct ShaderInfo {
	// Raised for expanded table calls: each inlined GT7 material variant brings its own
	// statically resolved descriptors.
	static constexpr uint32_t MaxBuffers      = 256;
	static constexpr uint32_t MaxImages       = 512;
	static constexpr uint32_t MaxSamplers     = 128;

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
	// The shader intersects rays with a BVH (BvhIntersect): its dispatches run once with the
	// pages they find, rather than retry until every node they reach is resident
	// (DispatchBufferRecovery).
	bool uses_bvh = false;

	bool operator==(const ShaderInfo& other) const = default;
};

struct BlockInfo {
	uint32_t        id       = 0;
	uint32_t        start_pc = 0;
	uint32_t        end_pc   = 0;
	CFG::Terminator terminator;
	Value           condition;
	Value           indirect_target;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct DescriptorSource {
	struct IndirectDescriptor {
		uint32_t material_source = UINT32_MAX;
		uint32_t table_source    = 0;
		uint32_t selector_stride = 0;
		uint32_t selector_offset = 0;
		uint32_t table_offset    = 0;
		uint32_t table_stride    = 0;
		uint32_t workgroup_axis  = UINT32_MAX;
		uint32_t selector_shift  = 0;
		// The key is (material word >> key_shift) & key_mask; identity unless the shader packs
		// two keys into one word.
		uint32_t key_shift = 0;
		uint32_t key_mask  = UINT32_MAX;
		// No enumeration: the shader looks the key up in the bindless translation table.
		bool     bindless  = false;
		// No table either: the T# is a value the shader loads; ImageResource::runtime.
		bool                  runtime = false;
		Value    key_count;
		Value                 selector_first;
		Value    selector_mask;
		std::vector<uint32_t> sources;

		bool operator==(const IndirectDescriptor& other) const = default;
	};

	// A sampler read from a sampler heap at a GPU-computed key: dwords 0-2 are the heap's V#
	// (dword 3 is left 0), the S# records start table_offset bytes into it.
	struct BindlessSampler {
		uint32_t table_offset = 0;

		bool operator==(const BindlessSampler& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectDescriptor> indirect_descriptor;
	std::optional<BindlessSampler>    bindless_sampler;

	bool operator==(const DescriptorSource& other) const = default;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct SrtRead {
	Value    value;
	uint32_t flat_offset = 0;

	bool operator==(const SrtRead& other) const = default;
};

// A store or atomic on a written buffer whose address operands the host can bound: immediates,
// values it evaluates, compute invocation IDs, and a few integer operations on them.
// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct BufferWrite {
	uint32_t buffer    = 0;
	uint32_t immediate = 0;
	Value    index;
	Value    offset;
	Value    soffset;
	// The store's lane predicate: a select on it takes the stored lanes' operand.
	Value    predicate;

	bool operator==(const BufferWrite& other) const = default;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct ResourceBlock {
	// Conditional successors are ordered true, false; an empty condition follows every edge.
	Value                 condition;
	std::vector<uint32_t> successors;
	std::vector<uint32_t> sources;
	std::vector<uint32_t> srt_reads;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct CompiledShaderInfo {
	ShaderType                    stage               = ShaderType::Unknown;
	uint64_t                      shader_hash         = 0;
	uint32_t                      wave_size           = 64;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	uint32_t                      param_export_mask   = 0;
	bool                          has_address_writes  = false;
	ShaderInfo                    info;
	BindingLayout                 bindings;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
};

// Resource analysis retained by the shader cache. It owns immutable descriptor/SRT,
// condition and fill values without translated blocks, plus reusable evaluation scratch.
class SrtNativeCode;
struct SrtTrace;

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct ResourcePlan {
	struct EvaluationContext {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};

		std::vector<Entry> values;
		uint64_t           generation = 0;
	};

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
	std::vector<MemoryInfo>             memory_info;
	std::vector<DescriptorSource>       descriptor_sources;
	std::vector<ResourceBlock>          control_flow;
	std::vector<SrtRead>                srt_reads;
	// Per buffer of info.buffers: every store to it is in buffer_writes (bounded write extent).
	std::vector<uint8_t>                buffer_writes_bounded;
	std::vector<BufferWrite>            buffer_writes;
	std::vector<uint8_t>                clean_flat_slots;
	// Flat slots only the shader uses (GpuFillSlots): a raw read whose address comes from user data
	// and immediates alone, which no host-evaluated value refers to.
	std::vector<uint8_t>                gpu_fill_slots;
	bool                                requires_specialization_memory = false;
	bool                                capture_specialization_reads = false;
	// The device supports bindless images: an indirect image the enumeration cannot cover
	// becomes a bindless one instead of failing tracking.
	bool                                bindless_images = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	ShaderInfo                          info;
	UniformFillPlan                     uniform_fill;
	// GPU-thread scratch for nested clean/EXEC memos, activity and material keys.
	mutable std::deque<EvaluationContext> evaluation_contexts;
	mutable uint32_t                       evaluation_value_count = 0;
	mutable uint32_t                       evaluation_depth       = 0;
	mutable std::vector<uint8_t>            active_sources;
	// active_sources before the walk (sources no block guards), built once per plan.
	mutable std::vector<uint8_t>            active_initial;
	mutable std::vector<uint8_t>            visited_blocks;
	mutable std::vector<uint32_t>           pending_blocks;
	mutable std::vector<uint32_t>           material_keys;
	// The plan compiled into x86-64 code (SrtNative.h) once it is refreshed often enough.
	// SrtWalker::RefreshFlatBuffer's last walk: the visited blocks that have a condition, with
	// its outcome (0 false, 1 true, 2 not known), the reads it evaluated (each once, in order),
	// the active sources it left and the reader setup it ran with. A walk whose conditions give
	// the same outcomes visits the same blocks.
	mutable std::vector<std::pair<uint32_t, uint8_t>> active_walk;
	mutable std::vector<uint8_t>                      active_walk_result;
	mutable std::vector<uint32_t>                     walk_slots;
	mutable std::vector<uint32_t>                     walk_slot_stamps;
	mutable uint32_t                                  walk_generation   = 0;
	mutable uint8_t                                   walk_key          = 0;
	mutable bool                                      active_walk_valid = false;
	mutable std::shared_ptr<const SrtNativeCode> native_code;
	mutable uint32_t                             native_uses      = 0;
	mutable bool                                 native_attempted = false;
	// The replay trace of the last refresh (SrtWalker.h, SrtTraceSession), and how recording went.
	mutable std::shared_ptr<SrtTrace> srt_trace;
	mutable uint32_t                  srt_trace_uses    = 0;
	mutable uint32_t                  srt_trace_misses  = 0;
	mutable uint32_t                  srt_trace_backoff = 0;
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
	uint32_t                      scratch_dwords = 0;
	bool                          dispatcher_fallback = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string                   fallback_reason;
	std::vector<BlockInfo>        block_info;
	struct ScalarWrite { uint32_t pc; ScalarReg reg; };
	std::vector<ScalarWrite>      scalar_writes;
	// Typed memory and export instructions reference shader-local metadata by dense index.
	// Decoder-only details (such as NSA register numbers) have already become IR operands.
	std::vector<ExportInfo>       export_info;
	bool                          has_address_writes = false;
	bool                          shader_info_complete = false;
	BindingLayout                 bindings;
	bool                          binding_layout_complete = false;

};

std::string ProgramToString(const Program& program);
bool        HasShaderMemoryWrites(const Program& program);

void  ValidateProgram(const Program& program, bool require_ssa);
// Whether the translation path validates the IR (debug builds, or KYTY_VALIDATE_IR=1).
bool  IrValidationEnabled();
void  ResolveControlFlowIdentities(Program& program);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);
Value ResolveActiveU32(Value value, Value active);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
