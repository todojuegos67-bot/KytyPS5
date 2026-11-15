#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Xbyak {
class CodeGenerator;
}

namespace Libs::Graphics::ShaderRecompiler::IR {

class SrtWalker;

// Finite descriptor domains also bound the arithmetic used to prove their indices.
constexpr uint32_t MaxIndirectDescriptorProbes = 65536u;

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

// Upstream 9a29dc62c: a null image takes the dimension and cube its instructions declare, and an image's cube flag
// comes from its instructions. Off by default (KYTY_NULL_IMAGE_DIMS=1 turns it on): it changes the keys of stages
// with a null image under a cube or 3D instruction, so Marvel's Wolverine's hints would need new recordings.
inline bool NullImageInstructionDims() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NULL_IMAGE_DIMS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
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
	bool                    address_is_full                                       = false;
	bool                    data_signed                                           = false;
	bool                    typed                                                 = false;
	bool                    formatted                                             = false;
	bool                    image_has_mip                                         = false;
	bool                    image_r128                                            = false;
	bool                    image_cube                                            = false;
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;

	[[nodiscard]] bool IsCoherentBufferPairLoad(ValueOpcode opcode) const {
		return kind == ResourceKind::Buffer && opcode == ValueOpcode::LoadBufferU32x2 && coherent &&
		       !typed && !formatted && data_bits == 32u;
	}

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		if (typed) return false;
		if (data_bits == 8u) return !formatted && opcode == ValueOpcode::LoadBufferU8;
		if (data_bits == 16u) return !formatted && opcode == ValueOpcode::LoadBufferU16;
		return data_bits == 32u &&
		       (formatted ? opcode == ValueOpcode::LoadBufferU32
		                  : opcode == ValueOpcode::ReadConstBuffer ||
		                        opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4);
	}
	// Raw DWORD stores through a V# known only at run time.
	[[nodiscard]] bool SupportsIndirectBufferStore(ValueOpcode opcode) const {
		return !formatted && !typed && data_bits == 32u &&
		       (opcode == ValueOpcode::StoreBufferU32 || opcode == ValueOpcode::StoreBufferU32x2 ||
		        opcode == ValueOpcode::StoreBufferU32x3 || opcode == ValueOpcode::StoreBufferU32x4);
	}
	[[nodiscard]] bool SupportsIndirectBufferAtomic(ValueOpcode opcode) const {
		if (formatted || typed || data_bits != 32u) return false;
		switch (opcode) {
			case ValueOpcode::BufferAtomicSwap32:
			case ValueOpcode::BufferAtomicCmpSwap32:
			case ValueOpcode::BufferAtomicIAdd32:
			case ValueOpcode::BufferAtomicISub32:
			case ValueOpcode::BufferAtomicSMin32:
			case ValueOpcode::BufferAtomicUMin32:
			case ValueOpcode::BufferAtomicSMax32:
			case ValueOpcode::BufferAtomicUMax32:
			case ValueOpcode::BufferAtomicAnd32:
			case ValueOpcode::BufferAtomicOr32:
			case ValueOpcode::BufferAtomicXor32: return true;
			default: return false;
		}
	}
	[[nodiscard]] bool SupportsIndirectBufferAccess(ValueOpcode opcode) const {
		return SupportsIndirectBufferLoad(opcode) || SupportsIndirectBufferStore(opcode) ||
		       SupportsIndirectBufferAtomic(opcode);
	}

	bool operator==(const MemoryInfo& other) const = default;
};

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind  = ExportTargetKind::Unknown;
	uint32_t         index = 0;
	uint32_t         en    = 0;
	bool             compr = false;
	bool             vm    = false;

	bool operator==(const ExportInfo& other) const = default;
};

struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               descriptor_index   = UINT32_MAX;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               write_extent       = 0; // UINT32_MAX once a write is runtime-addressed or formatted
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	uint32_t               image_alias        = NoImageAlias;
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	bool                   live               = false;
	// Written where no store bound is known (MarkTrackedWrites): a dispatch also records the pages it
	// writes in a bitmap, so only those stay GPU-written once the GPU has finished it.
	bool                   tracked            = false;
	uint32_t               indirect_root              = NoIndirectBuffer;
	uint32_t               indirect_mapping_offset    = 0;
	uint32_t               indirect_search_iterations = 0;
	std::vector<uint32_t>  indirect_resources;
	// This buffer is a table of V#s (16-byte entries starting at indirect_table_offset) that
	// selects the targets of IndirectBuffer stores; the host caches every entry it names.
	bool                   indirect_write_table  = false;
	uint32_t               indirect_table_offset = 0;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;
// A form slot's form in the shader data's form dword (passes/ImageForms.h): this many bits a slot.
constexpr uint32_t ImageFormBits = 3u;
// A specialization's form image (passes/ImageForms.h) has this mapping offset and its slot as its search iterations: the
// fields of a table's root, which a form image has no use for, so that a specialization keeps its size and its bytes.
constexpr uint32_t ImageFormMapping = UINT32_MAX;

// A field the renderer's ResolveTexture reads also belongs in RenderExecutor::TextureMemoResource, its memo's key.
struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;
	static constexpr uint32_t NoFormSlot      = UINT32_MAX;

	uint32_t                      source            = 0;
	uint32_t                      descriptor_index  = UINT32_MAX;
	uint32_t                      first_use_pc      = 0;
	ImageResourceClass            resource_class    = ImageResourceClass::None;
	Prospero::TextureNumericClass numeric_class     = Prospero::TextureNumericClass::Unsupported;
	Decoder::ImageDimension       dimension         = Decoder::ImageDimension::Unknown;
	ImageMipMode                  mip_mode          = ImageMipMode::None;
	uint32_t                      mip_count         = 1;
	Prospero::BufferFormat        conversion_format = Prospero::BufferFormat::kInvalid;
	uint32_t                      shader_swizzle    = ShaderImageIdentitySwizzle;
	Prospero::BufferFormat        srgb_decode       = Prospero::BufferFormat::kInvalid;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          atomic64          = false;
	bool                          unfiltered_access_only = true;
	bool                          sample_only       = true; // sampled, gathered or LOD-queried: no integer coordinates
	// Sampled, loaded (the emitter clamps a fetch's layer) or size-queried without the layer count: a 2D texture under
	// an array instruction reads the same as its one-layer array view.
	bool                          one_layer_safe    = true;
	// Only loaded or stored without a mip operand: every access is to the view's base level, so a view of
	// that one level reads and writes the same as a view of all of them.
	bool                          base_level_only   = true;
	bool                          depth_compare     = false;
	bool                          skip_degamma      = false; // A sampler reads sRGB as stored.
	bool                          cube              = false;
	bool                          r128              = false;
	bool                          heap              = false; // Resident image heap root or kind.
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	// A group of an indirect storage root's same-typed candidates (GroupIndirectImages): an array of
	// this many descriptors, the draw's candidates first and null descriptors after them. 0: one.
	uint32_t                      indirect_capacity          = 0;
	std::vector<uint32_t>         indirect_resources;
	// Read from the game's descriptor heap by slot (the guest image heap, --bindless): the image is
	// element `slot` of that heap's array for this image's binding kind, and has no descriptor of
	// the shader's own.
	bool                          guest_heap        = false;
	// A form slot (passes/ImageForms.h): the image is the root of its forms, its candidates one form each in the
	// order of the title's forms, and the shader data's form dword says which one the dispatch binds.
	uint32_t                      form_slot         = NoFormSlot;

	bool operator==(const ImageResource& other) const = default;
};

[[nodiscard]] constexpr Prospero::BufferFormat
RawSignedStorageFormat(const ImageResource& image, Prospero::BufferFormat format) {
	if (image.resource_class != ImageResourceClass::Storage || !image.written)
		return Prospero::BufferFormat::kInvalid;
	if (format == Prospero::BufferFormat::k32SInt && (image.atomic ? !image.atomic64 : !image.read))
		return Prospero::BufferFormat::k32UInt;
	if (format == Prospero::BufferFormat::k16SInt && !image.read && !image.atomic)
		return Prospero::BufferFormat::k16UInt;
	return Prospero::BufferFormat::kInvalid;
}

// A written storage image of a 16-bit packed UNorm format. Vulkan requires no GPU to store to these and NVIDIA's do
// not (VUID-VkImageViewCreateInfo-usage-02275), so the host binds an R16_UINT view of the image and the shader packs
// and unpacks the texels in the guest's bit layout (conversion_format), on every GPU. k1_5_5_5UNorm is not a texture
// format here (gpu_format.cpp), so it is not one either.
[[nodiscard]] constexpr Prospero::BufferFormat
PackedStorageFormat(const ImageResource& image, Prospero::BufferFormat format) {
	if (image.resource_class != ImageResourceClass::Storage || !image.written)
		return Prospero::BufferFormat::kInvalid;
	switch (format) {
		case Prospero::BufferFormat::k5_6_5UNorm:
		case Prospero::BufferFormat::k5_5_5_1UNorm:
		case Prospero::BufferFormat::k4_4_4_4UNorm: return Prospero::BufferFormat::k16UInt;
		default: return Prospero::BufferFormat::kInvalid;
	}
}

// A raw load of a kDepth k16UInt T# reads a D16 surface's codes. Vulkan cannot view D16 as
// R16Uint, so the host binds it as UNorm and the shader converts back.
[[nodiscard]] constexpr bool ReadsDepthBits(const ImageResource& image) {
	return image.conversion_format == Prospero::BufferFormat::k16UInt;
}

struct SamplerResource {
	static constexpr uint32_t NoIndirectSampler = UINT32_MAX;

	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	// Native filtering/border variants share the original sampler's runtime descriptor.
	uint32_t snapshot_index        = 0;
	uint32_t indirect_root = NoIndirectSampler;
	uint32_t indirect_mapping_offset = 0;
	uint32_t indirect_search_iterations = 0;
	std::vector<uint32_t> indirect_resources;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;
	// Read from the game's sampler heap by key: element key >> 4 of the guest image heap's sampler
	// array for its class (ImageHeapFloatSamplerBinding, ImageHeapIntegerSamplerBinding).
	bool     guest_heap            = false;

	bool operator==(const SamplerResource& other) const = default;
};

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
	PatchInput,
	Factor
};
constexpr uint32_t NumTessellationAttributes =
    static_cast<uint32_t>(TessellationAttribute::Factor) + 1u;

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
	BaryCoordSmoothSample,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	NumWorkgroups,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	BaseVertex,
	BaseInstance,
	Parameter,
};

enum class StageOutputKind {
	Position,
	Parameter,
	Mrt,
	ObjectPrimitiveId,
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
inline constexpr uint32_t ImageBindingCount           = 50u;
// Resident image heaps share one sampled-image array at binding 0 of this set. A heap slot
// word is kind << ImageHeapKindShift | element; element 0 is the null descriptor.
inline constexpr uint32_t ImageHeapSet                = 1u;
inline constexpr uint32_t ImageHeapCapacity           = 65536u;
inline constexpr uint32_t ImageHeapKindShift          = 20u;
// An indirect storage root with this many candidates or more has them grouped by type
// (GroupIndirectImages): its table words map to group << IndirectImageGroupShift | rank, group 0
// being the root itself (the null descriptor).
inline constexpr uint32_t IndirectImageGroupMin       = 17u;
inline constexpr uint32_t IndirectImageGroupShift     = 20u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	SharedMemory,
	// Read-only words the emitter made for the shader itself (CompiledShaderInfo::program_constants).
	ProgramConstants,
	Count,
};

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 51u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 59u);

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 5;
	// Upstream abc509d33: indirect mesh draws read the draw parameters, then the record's first instance, from the
	// task payload (gpu_mesh_indirect.task).
	static constexpr uint32_t MeshTaskPayloadDwordCount = MeshDrawDwordCount + 1;
	static constexpr uint32_t NoStart    = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

	// User clip planes 0..N follow the mesh draw parameters or object ID, four dwords each.
	static constexpr uint32_t UserClipPlaneDword = MeshDrawDwordCount;

	[[nodiscard]] static constexpr uint32_t GraphicsDrawDwords(bool mesh, bool object_id,
	                                                           uint32_t user_clip_planes = 0) {
		if (user_clip_planes != 0) {
			return UserClipPlaneDword + 4u * std::bit_width(user_clip_planes);
		}
		return mesh ? MeshDrawDwordCount : static_cast<uint32_t>(object_id);
	}

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

[[nodiscard]] constexpr uint32_t ImageBindingIndex(DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) - FirstImageBinding;
}

[[nodiscard]] constexpr std::optional<DescriptorBindingKind>
DescriptorBindingForImage(const ImageResource& image) {
	constexpr uint32_t SampledFloatBinding = 1u;
	constexpr uint32_t SampledUintBinding  = 8u;
	constexpr uint32_t SampledSintBinding  = 15u;
	constexpr uint32_t StorageFloatBinding = FirstStorageImageBinding;
	constexpr uint32_t StorageUintBinding  = StorageFloatBinding + 6u;
	constexpr uint32_t AtomicUintBinding   = StorageUintBinding + 6u;

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
			if (image.atomic) {
				return std::nullopt;
			}
			dimension = sampled ? 4u : 5u;
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

// The guest image heap (--bindless) is descriptor set 1 (ImageHeapSet), the same for every stage, in
// place of the resident image heaps above: the game's descriptor heap, a slot an element. A sampled image binding kind is binding ImageBindingIndex(kind) there. Binding
// ImageHeapInfoBinding holds a dword a slot: the numeric class of the slot's texture (bits 0-1:
// float, uint, sint), whether it is a cube, a bit for each dimension it can be read as, and
// (bits 8-11) the level of the texture that is level 0 of what the host holds: the host holds a
// streamed texture from its first loaded level on. A slot with no dimension bit is read as slot
// 0, which holds a null image.
// A pixel shader's own resources with graphics pipeline libraries (GraphicContext::SplitsFragmentSet): its library
// is built without the vertex stages, so their sets cannot be one.
inline constexpr uint32_t FragmentSet          = 2u;
// A program's image groups (ImageResource::indirect_capacity) in a set of their own: vertex and compute programs' in
// GroupSet, pixel programs' in GroupSet + 1 (a library holds one stage). A draw whose groups haven't changed binds the
// set it had instead of writing thousands of candidates again (RenderExecutor::CommitBindings). Other stages keep
// their groups in their main set.
inline constexpr uint32_t GroupSet             = 3u;
inline constexpr uint32_t NoGroupSet           = UINT32_MAX;
[[nodiscard]] constexpr uint32_t GroupSetFor(ShaderType stage) {
	return stage == ShaderType::Pixel                                     ? GroupSet + 1u
	       : stage == ShaderType::Vertex || stage == ShaderType::Compute ? GroupSet
	                                                                     : NoGroupSet;
}
inline constexpr uint32_t ImageHeapSlots       = 65536u;
inline constexpr uint32_t ImageHeapInfoBinding = ImageBindingCount;
inline constexpr uint32_t ImageHeapCube        = 4u;
inline constexpr uint32_t ImageHeapLevelShift  = 8u;
// The game's sampler heap (16 bytes a record) in the same set: each record as a float sampler and
// as an integer one (point filtering, integer border), as the shader's own samplers are made.
inline constexpr uint32_t ImageHeapFloatSamplerBinding   = ImageHeapInfoBinding + 1u;
inline constexpr uint32_t ImageHeapIntegerSamplerBinding = ImageHeapInfoBinding + 2u;
inline constexpr uint32_t ImageHeapSamplerSlots          = 256u;

[[nodiscard]] constexpr uint32_t ImageHeapFilled(Decoder::ImageDimension dimension) {
	switch (dimension) {
		case Decoder::ImageDimension::Dim2D: return 8u;
		case Decoder::ImageDimension::Dim2DArray: return 16u;
		case Decoder::ImageDimension::Dim3D: return 32u;
		case Decoder::ImageDimension::Dim1D: return 64u;
		default: return 0u;
	}
}

struct Bindings {
	std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)> descriptor_counts {};
	uint64_t descriptor_mask = 0;
	// Group arrays in the group set (GroupSetFor), by ImageBindingIndex: their elements, none in descriptor_counts.
	std::array<uint32_t, ImageBindingCount> group_counts {};
	uint32_t push_data_start_dword = PushData::NoStart;
	uint32_t dispatch_thread_dword = PushData::NoStart;
	uint32_t memory_offset_dword = 0;

	// Past the packed byte offsets, each buffer's bound range in bytes: the shader's bounds. Not
	// OpArrayLength, which NVIDIA reports as 0 for a range of 2 GiB or more (within its 4 GiB limit).
	[[nodiscard]] uint32_t BufferLengthDword() const {
		return memory_offset_dword +
		       (descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)] + 3u) / 4u;
	}
	// With a tracked buffer (BufferResource::tracked), past the lengths each buffer's write bitmap
	// address, two dwords (0: not tracked in this dispatch).
	bool write_tracking = false;
	[[nodiscard]] uint32_t TrackDword() const {
		return BufferLengthDword() + descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)];
	}
	// With form slots (ImageResource::form_slot), past the tracking words the dispatch's forms, one dword.
	bool image_forms = false;
	[[nodiscard]] uint32_t ImageFormDword() const {
		return TrackDword() +
		       (write_tracking ? 2u * descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)] : 0u);
	}
	// the buffer table variant has buffers but no storage-buffer binding, and past the
	// forms each buffer's base address, two dwords (the bound range's first byte, as its descriptor named it).
	[[nodiscard]] bool BufferTable() const {
		return descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)] != 0 &&
		       (descriptor_mask & (uint64_t {1} << static_cast<uint32_t>(DescriptorBindingKind::Buffers))) == 0;
	}
	[[nodiscard]] uint32_t TableDword() const {
		return ImageFormDword() + (image_forms ? 1u : 0u);
	}
	[[nodiscard]] uint32_t ShaderDataDwords() const {
		return TableDword() +
		       (BufferTable() ? 2u * descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)] : 0u);
	}
	[[nodiscard]] bool UsesPushData() const {
		return push_data_start_dword != PushData::NoStart;
	}
	void AdvancePushData(uint32_t& cursor) const {
		if (UsesPushData()) {
			cursor = push_data_start_dword + ShaderDataDwords();
		}
	}

	bool operator==(const Bindings& other) const = default;
};

struct ShaderInfo {
	static constexpr uint32_t MaxBufferRoots  = 64;
	static constexpr uint32_t MaxImages       = 64;
	static constexpr uint32_t MaxSamplers     = 32;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	std::vector<std::array<uint32_t, 2>> address_bases; // Flattened SRT slots of global bases.
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::vector<uint32_t>            user_data_registers;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                            has_bitwise_xor               = false;
	bool                            uses_dma                      = false;
	bool                            uses_swizzle                  = false;
	bool                            uses_lds                      = false;
	bool                            uses_gds                      = false;
	bool                            uses_flattened_srt            = false;
	bool                            uses_image_heap               = false;
	bool                            uses_device_address           = false;
	bool                            uses_dispatch_threads         = false;
	bool                            bvh                           = false;
	bool                            bvh8                          = false;
	bool                            bvh_stack                     = false;
	bool                            subgroup_ballot               = false;
	bool                            subgroup_vote                 = false;
	bool                            subgroup_barrier              = false;
	bool                            subgroup_shuffle              = false;
	bool                            subgroup_arithmetic           = false;
	bool                            subgroup_local_invocation_id  = false;
	bool                            helper_invocation             = false;
	bool                            compute_derivatives           = false;
	bool                            image_gather_extended         = false;
	bool                            function_lds                  = false;
	uint32_t                        function_lds_dwords           = 0; // what its accesses can reach
	bool                            function_scratch              = false;
	bool                            pixel_valid_mask              = false;
	bool                            buffer_int64_atomics          = false;
	bool                            buffer_u8                     = false;
	bool                            buffer_u16                    = false;
	bool                            shared_int64_atomics          = false;
	bool                            coherent_buffers              = false;
	bool                            float64                       = false;
	// Compute stores through V#s known only at run time: they write by device address (uses_dma).
	bool                            indirect_stores               = false;
	// User-data registers of the pointers (low, high) those V#s' addresses derive from.
	std::vector<std::pair<uint32_t, uint32_t>> indirect_store_roots;

	bool operator==(const ShaderInfo& other) const = default;
};

struct DescriptorSource {
	struct IndirectDescriptor {
		struct SelectorRead {
			struct CountedRows {
				uint32_t source = UINT32_MAX;
				uint32_t stride = 0; // Byte spacing between selector rows, not their length.

				bool operator==(const CountedRows& other) const = default;
			};
			struct MaskRanks {
				uint32_t source = UINT32_MAX;
				uint32_t stride = 0; // Byte spacing of the mask words its load can address.
				uint32_t offset = 0;

				bool operator==(const MaskRanks& other) const = default;
			};
			uint32_t source = UINT32_MAX;
			uint32_t stride = 0;
			uint32_t offset = 0;
			// V#-strided reads select records first_record + k * record_step.
			uint32_t first_record = 0;
			uint32_t record_step = 1;
			uint32_t maximum = UINT32_MAX;
			bool non_negative = false; // Signed-negative fields read as zero before the maximum.
			uint32_t bit_offset = 0;
			uint32_t bit_mask = UINT32_MAX;
			bool global = false;
			// Four or eight DWORD bitmap words per record; set ranks subtract from the last key.
			bool bitset = false;
			std::optional<CountedRows> rows;
			std::vector<std::pair<uint32_t, uint32_t>> byte_ranges;
			std::optional<uint32_t> excluded; // A guard keeps this key from the read.
			std::optional<MaskRanks> ranks;   // Records are the bits some mask word sets.

			bool operator==(const SelectorRead& other) const = default;
		};
		struct PointerTable {
			uint32_t offset = 0;
			uint32_t stride = 0;
			uint32_t count = 1;
			bool indexed = false; // Parent pointers use raw MUBUF record addressing.

			bool operator==(const PointerTable& other) const = default;
		};
		std::optional<SelectorRead> selector;
		std::optional<uint32_t> table_maximum;
		std::optional<PointerTable> pointer_table;
		uint32_t table_source    = 0;
		uint32_t table_offset    = 0;
		uint32_t table_immediate = 0;
		uint32_t table_stride    = 0;
		uint32_t table_record_bytes = 0;
		bool     table_scalar = false;
		bool     indexed = false;
		bool     signed_count = false;
		uint32_t workgroup_axis  = UINT32_MAX;
		// Nonzero shift: workgroup keys are (selector_first + ID) * multiplier >> shift.
		uint32_t workgroup_multiplier = 0;
		uint32_t workgroup_shift      = 0;
		Value    key_count;
		// First selector index, workgroup key base, or first descriptor key for a direct or bitmap
		// range.
		Value                 selector_first;
		Value    selector_mask;
		std::vector<uint32_t> sources;

		bool operator==(const IndirectDescriptor& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectDescriptor> indirect_descriptor;

	bool operator==(const DescriptorSource& other) const = default;
};

// Where a table-read sampler's record lies in the game's sampler heap, from the key the shader passes. The key is
// the table's record selector, shifted left by 2 when its maximum is known (ResourceTracking), not a byte offset:
// the shader reads the S# at ((selector * stride + offset) & ~3) + (immediate & ~3), as the candidate path does.
// With a stride of whole 16-byte records and a first record on one, that is record
// (selector * stride + base) >> 4 in U32 arithmetic. Empty for a table whose records the heap cannot hold whole.
struct HeapSamplerRecord {
	uint32_t selector_shift = 0;
	uint32_t stride         = 0;
	uint32_t base           = 0;

	[[nodiscard]] constexpr uint32_t Slot(uint32_t key) const {
		return ((key >> selector_shift) * stride + base) >> 4u;
	}
	bool operator==(const HeapSamplerRecord& other) const = default;
};

[[nodiscard]] inline std::optional<HeapSamplerRecord>
HeapSamplerRecordOf(const DescriptorSource::IndirectDescriptor& indirect) {
	const auto base = (indirect.table_offset & ~3u) + (indirect.table_immediate & ~3u);
	if ((!indirect.selector && !indirect.table_maximum) || indirect.table_stride == 0u ||
	    (indirect.table_stride & 15u) != 0u || (base & 15u) != 0u) {
		return std::nullopt;
	}
	return HeapSamplerRecord {.selector_shift = indirect.table_maximum ? 2u : 0u,
	                          .stride         = indirect.table_stride,
	                          .base           = base};
}

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
	std::vector<uint32_t> images;
	std::vector<uint32_t> srt_reads;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
struct CompiledShaderInfo {
	ShaderType                    stage               = ShaderType::Unknown;
	uint64_t                      shader_hash         = 0;
	uint32_t                      wave_size           = 64;
	uint32_t                      user_data_base      = 0;
	uint32_t                      param_export_mask   = 0;
	bool                          has_address_writes  = false;
	bool                          wave_split          = false; // Program::wave_split: needs full subgroups
	ShaderInfo                    info;
	Bindings                      bindings;
	// What its ProgramConstants binding holds, the same for every draw. Shared, so that the host's buffer of
	// it (BufferCache::GetProgramConstants) can know it by address.
	std::shared_ptr<const std::vector<uint32_t>> program_constants;
};

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
	Value                pattern_size;
	Value                element_count;
};

// The last materialization of each indirect image table a plan reads whole (ResourceMaterialization.cpp).
struct IndirectImageReuse;
struct IndirectImageReuseDelete {
	void operator()(IndirectImageReuse* reuse) const;
};

// Resource analysis retained by the shader cache. It owns immutable descriptor/SRT,
// condition and fill values without translated blocks, plus reusable evaluation scratch.
struct ResourcePlan {
	struct SelectorKeys {
		uint32_t              source = UINT32_MAX;
		std::vector<uint32_t> values;
		bool                  resolved = false;
	};

	struct EvaluationContext {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};

		std::vector<Entry> values;
		uint64_t           generation = 0;
		const SrtWalker*    session = nullptr;
		Value              active_mask;
		uint8_t            read_policy = 0;
	};

	// One interpreted RefreshFlatBuffer as a flat list of the evaluations it made, in order
	// (SrtWalker.cpp, "Replay traces"). A ref is an op (>= 0), an immediate (-1 - index) or
	// TraceFailed.
	struct TraceOp {
		uint8_t     kind    = 0;
		uint8_t     flags   = 0;
		uint8_t     count   = 0;     // args used
		bool        ok      = false; // what the interpreter got
		ValueOpcode opcode  = ValueOpcode::Void;
		uint32_t    imm     = 0;
		uint64_t    value   = 0;     // the interpreter's result, for results fixed by the checks
		std::array<int32_t, 5> args {};
	};
	struct Trace {
		struct Seed {
			uint32_t index   = 0;
			int32_t  ref     = 0;
			uint8_t  context = 0; // 0 the walker's, 1 its clean evaluator's
			bool     unknown = false;
		};
		std::vector<TraceOp>  ops;
		std::vector<uint64_t> immediates;
		std::vector<Seed>     seeds;
		std::vector<uint8_t>  visited;
		std::vector<uint8_t>  active;
		std::vector<uint8_t>  active_images;
		uint8_t               key   = 0;
		bool                  valid = false;
	};

	ResourcePlan();
	~ResourcePlan();

	ResourcePlan(const ResourcePlan&)            = delete;
	ResourcePlan& operator=(const ResourcePlan&) = delete;
	ResourcePlan(ResourcePlan&&) noexcept;
	ResourcePlan& operator=(ResourcePlan&& other) noexcept;

	ShaderType                    stage           = ShaderType::Unknown;
	uint64_t                      shader_hash     = 0;
	uint32_t                      user_data_base  = 0;
	uint32_t                      user_data_count = 64;
	std::list<Inst>                     value_storage;
	std::vector<MemoryInfo>             memory_info;
	std::vector<DescriptorSource>       descriptor_sources;
	std::vector<uint32_t>               selector_groups;
	std::vector<ResourceBlock>          control_flow;
	std::unordered_map<const Inst*, std::vector<uint32_t>> control_phi_predecessors;
	std::vector<SrtRead>                srt_reads;
	// Per SRT read: 1 when a value the host evaluates depends on it (a descriptor word, a branch condition, the
	// uniform fill, a buffer store's bound, a global base, another read's address); 0 when only the shader reads it,
	// from the flat buffer. Derived on load (MarkSrtHostReads).
	std::vector<uint8_t>                srt_read_host;
	// How SrtWalker::EvaluateDescriptor fetches a descriptor dword after the refresh instead of evaluating it: the
	// flattened slot of the SRT read it is, a user-data word, or an immediate. shadPS4 (SharpFetch, PR #5112) fixes
	// each sharp's dwords to flat-buffer slots when it compiles a shader; ExtractResourcePlan does the same. Empty, or
	// Walk, evaluates the dword.
	struct DescriptorFetch {
		enum Kind : uint8_t { Walk, Flat, UserData, Immediate };
		Kind        kind  = Walk;
		uint32_t    index = 0;          // Flat: the flat slot; UserData: the word; Immediate: the value
		uint32_t    block = UINT32_MAX; // Flat in a plan with control flow: the one block that refreshes the read
		const Inst* read  = nullptr;    // Flat: the read
	};
	std::vector<std::array<DescriptorFetch, 8>> descriptor_fetch; // by descriptor source
	bool                                requires_specialization_memory = false;
	std::vector<std::pair<uint64_t, uint64_t>> source_reads;
	bool                                capture_specialization_reads = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	ShaderInfo                          info;
	UniformFillPlan                     uniform_fill;
	// A compute shader's stores and atomics on written buffers whose addresses the host can
	// bound (index, VGPR offset, SGPR offset), with the parts of them it evaluates. A buffer
	// with any other store has none here: it counts as written whole.
	struct BufferStore {
		uint32_t             buffer    = 0;
		uint32_t             immediate = 0;
		std::array<Value, 3> address;
		std::vector<Value>   values;
		Value                guard; // its lane predicate (see ResolveActiveU32)
		bool                 per_lane = false; // its address needs each invocation's own loads
	};
	std::vector<BufferStore>            buffer_stores;
	// x86 walker over the retained SRT, descriptor and condition values (upstream d7856753e), for runtimes
	// whose guest loads give the readers' bytes (SrtRuntime::direct_reads).
	std::unique_ptr<Xbyak::CodeGenerator> srt_code;
	uint32_t                              srt_user_data = 0; // User-data DWORDs the walker reads.
	uint32_t                              srt_captures  = 0; // Scratch offset of loaded addresses.
	uint32_t                              srt_outputs   = 0; // Scratch bytes of descriptors, activity and outputs.
	uint32_t                              srt_reach     = 0; // Scratch offset of the reach byte per block.
	// GPU-thread scratch for walker results; srt_loads: the last walk's loaded addresses (at srt_captures);
	// srt_walked: the last materialization took the walk.
	mutable std::vector<uint64_t>           srt_scratch;
	// By Inst::EvaluationIndex: the value's walker output (UINT32_MAX: none). Upstream keeps it in the Inst,
	// which lean's interpreter index fills to its size limit.
	mutable std::vector<uint32_t>           srt_output_index;
	mutable uint32_t                        srt_loads  = 0;
	mutable bool                            srt_walked = false;
	// Per SRT read of a compute plan: the scratch offset where the walk keeps its shader-only load's address
	// (UINT32_MAX: none). Derived on load (CompileSrtWalker).
	std::vector<uint32_t>                   srt_fill_address;
	// The last walk's flat words left for the GPU to copy (SetSrtGpuFills): guest address, flat index.
	mutable std::vector<std::pair<uint64_t, uint32_t>> srt_fills;
	// GPU-thread scratch for nested clean/EXEC memos, activity and material keys.
	mutable std::deque<EvaluationContext> evaluation_contexts;
	mutable uint32_t                       evaluation_value_count = 0;
	mutable std::vector<uint8_t>            active_sources;
	// Per image: whether a reached block uses it (empty: all are), like active_sources.
	mutable std::vector<uint8_t>            active_images;
	mutable std::vector<uint8_t>            visited_blocks;
	mutable std::vector<uint8_t>            reachable_blocks;
	mutable std::vector<uint32_t>           pending_blocks;
	mutable std::vector<uint32_t>           material_keys;
	mutable std::vector<SelectorKeys>       selector_keys;
	// The last interpreted refresh, replayed by the next ones while its checks hold.
	bool                                    srt_trace = true; // false: interpret every refresh (tests)
	mutable Trace                           trace;
	mutable uint32_t                        trace_misses  = 0;
	mutable uint32_t                        trace_skip    = 0;
	mutable uint64_t                        trace_replays = 0; // refreshes served by the trace
	mutable std::vector<uint64_t>           trace_values;
	mutable std::vector<uint8_t>            trace_ok;
	// Image tables read whole, with the candidates and groups they made, reused while a table is unchanged.
	mutable std::unique_ptr<IndirectImageReuse, IndirectImageReuseDelete> image_reuse;
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
	// A wave64 program emitted one guest lane per invocation, each wave over two full 32-wide
	// subgroups (Profile::wave64_split), and the blocks where BranchPerInvocation lets its
	// invocations diverge: no wave-wide exchange may run there.
	bool                                 wave_split = false;
	std::unordered_set<const Block*>     wave_split_divergent_blocks;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string                   fallback_reason;
	struct ScalarWrite { uint32_t pc; ScalarReg reg; Value value; };
	std::vector<ScalarWrite>      scalar_writes;
	// Typed memory and export instructions reference shader-local metadata by dense index.
	// Decoder-only details (such as NSA register numbers) have already become IR operands.
	std::vector<ExportInfo>       export_info;
	bool                          has_address_writes = false;
	bool                          shader_info_complete = false;
	Bindings                      bindings;
	std::vector<uint32_t>         program_constants; // CompiledShaderInfo::program_constants

};

uint32_t StorageBufferElementBits(const Program& program, const MemoryInfo& memory);

std::string ProgramToString(const Program& program);
bool        HasShaderMemoryWrites(const Program& program);

void  ValidateProgram(const Program& program);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
bool  EquivalentSelectorDomain(const ResourcePlan& program,
                               const DescriptorSource::IndirectDescriptor& left,
                               const DescriptorSource::IndirectDescriptor& right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);
Value ResolveActiveU32(Value value, Value active);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
