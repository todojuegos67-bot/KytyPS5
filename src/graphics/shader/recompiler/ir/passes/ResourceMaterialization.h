#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		Prospero::BufferFormat        srgb_decode                = Prospero::BufferFormat::kInvalid;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		uint32_t                      indirect_capacity          = 0; // a group's array (GroupIndirectImages)
		bool                          cube                       = false;
		bool                          skip_degamma               = false;
		bool                          fmask                      = false;
		bool                          heap                       = false; // resident image heap (ImageHeapKind)
		bool                          guest_heap                 = false; // guest image heap slot (--bindless)
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool     guest_heap = false;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;
	// storage buffers by address from shader data (Bindings::BufferTable).
	bool                 buffer_table = false;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);
// An extracted plan's bytes for the stage store (Serialize.h), and back: a loaded plan is the extracted one, its
// derived state (descriptor fetches, walker, selector keys) made again. Empty: a value the bytes cannot hold. False:
// bytes of another layout or damaged.
std::vector<uint8_t> SaveResourcePlan(const ResourcePlan& plan);
bool                 LoadResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan);
// Marks the written buffers of a compute shader whose stores have no bound (BufferResource::tracked).
void MarkTrackedWrites(Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// walked, when non-null, is the flat buffer of an x86 walk of this runtime that the plan's scratch still
// holds (SrtReadLog::Repeats walked it and found it changed): it is taken instead of walking again.
// A refresh sets ResourceSpecialization::buffer_table to BufferTableWanted.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          const std::vector<uint32_t>* walked = nullptr);
// Whether a program takes the buffer table variant: it has storage buffers, and KYTY_TABLE_OFF=1 is not set (the way
// back to buffers by descriptor).
bool BufferTableWanted(const ResourcePlan& program);
// Tests: this thread computes a per-lane store's ranges with the per-lane interpreter (the reference) instead of the
// fast path. Both give the same ranges from the same reads.
void SetLaneWriteRangesReference(bool on);
// One candidate V# of an indirect buffer table, as `resource` reads it (indexed: by record index): true and the value
// kept, or zeroed for a record that reads nothing; false for a record the specialization cannot represent exactly.
bool NormalizeIndirectBuffer(DescriptorValue& value, bool indexed, const BufferResource& resource);

// Groups the candidates of each indirect storage root with IndirectImageGroupMin or more by type:
// one image per type, an array of a power-of-two capacity, replaces them, and the root's table
// words map to group and rank instead of ordinal. The specialization then depends on the types
// only, not on which descriptors the table holds (a game's image heap changes as it streams).
void GroupIndirectImages(const ShaderInfo& base, ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// After ApplyResourceSpecialization: the resource plan evaluates bound descriptors on the host, so their handles'
// dwords become 0. A handle the GPU still selects by key keeps that key (its first argument): an indirect buffer or
// image root, a sampler with candidates, and an image or sampler read from the game's heap by key.
void DropBoundDescriptorWords(Program& program);

// A game heap entry's guest image heap info dword (ShaderIR.h): 0 when the host shows nothing there.
uint32_t HeapImageInfo(const DescriptorValue& descriptor, bool r128);

// One changed entry of the game's descriptor heap: its slot and its 32 bytes now.
struct ImageHeapChange {
	uint32_t                slot = 0;
	std::array<uint32_t, 8> descriptor {};
};

// Compares the game's heap (`guest`, 8 dwords a slot) with `mirror`, the copy of the last call,
// and brings the copy up to date. Slot 0 is the game's reserved slot.
std::vector<ImageHeapChange> DiffImageHeap(std::span<const uint32_t> guest, std::vector<uint32_t>& mirror);

// Kind of an image heap entry as root sees it, or 0 (and a zeroed value) when root reads it as
// null. Entries of one kind specialize identically.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
// The sampled resource that root resolves entries of kind as; kind 0 gives the identity of
// root's view of a heap.
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
