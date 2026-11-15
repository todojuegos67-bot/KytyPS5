#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "common/groupReuse.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<TextureBinding>           images;
	// Per image: the fill of the texture memo's entry that `images` holds a copy of (ImageSlotMemo), 0 for none.
	std::vector<uint64_t>                 image_fills;
	// Per image: a group's candidates in rank order (an image with indirect_capacity); empty otherwise.
	std::vector<std::vector<TextureBinding>> image_groups;
	// Per image, beside its group: what its bindings were resolved from (PrepareBindings, Common::ResolveGroup), the
	// lookups its Held candidates repeat, the ranks bound in this draw, and its array's image infos as last committed
	// with the ranks to write again (all of them while `infos_whole`).
	struct GroupLookup {
		ImageId  lookup;
		uint64_t stamp               = 0;
		uint32_t metadata_base_layer = 0;
		uint64_t address             = 0; // the bytes no target of the draw may overlap
		uint64_t size                = 0;
	};
	struct Group {
		Common::GroupReuse<ShaderRecompiler::IR::DescriptorValue, ShaderRecompiler::IR::ImageResource> reuse;
		std::vector<GroupLookup>             lookups;
		std::vector<uint32_t>                bound;
		std::vector<vk::DescriptorImageInfo> infos;
		std::vector<uint32_t>                dirty;
		bool                                 infos_whole = true;
		// A new number (from one counter) whenever `infos` changes: a kept group set written from the same numbers
		// holds what `infos` holds now (RenderExecutor::CommitBindings).
		uint64_t                             generation  = 0;
		// A store program's candidates with those out of its reach made null (WolverineStores::Narrow), and whether
		// this draw's bindings were resolved from them.
		std::vector<ShaderRecompiler::IR::DescriptorValue> narrowed;
		std::vector<uint32_t>                 narrowed_kept;
		bool                                  narrowed_now = false;
		// The state the last walk repeated every Held candidate's lookup under (TextureCache::LookupState).
		TextureCache::LookupState            lookups_held;
		// What lets a draw's walks leave out candidates with nothing to do (GroupNotices). The ranks that resolved in this
		// draw; the bound ranks by image id, which the state `settled` says are still those images, and every one of
		// them visited at that state; the ranks that aren't storage, which every transit visits; the ranks whose image
		// the last transit found a target, whose notice isn't used up; and how far each walk has read the notices.
		std::vector<uint32_t>                 resolved;
		std::vector<std::pair<uint32_t, uint32_t>> image_ranks;
		std::vector<uint32_t>                 always;
		std::vector<uint32_t>                 held_targets;
		std::vector<uint32_t>                 visits;
		TextureCache::LookupState             settled;
		uint64_t                              rebind_notices  = 0;
		uint64_t                              transit_notices = 0;
		bool                                  settled_valid   = false;
		bool                                  transit_valid   = false;
	};
	std::vector<Group> groups;
	// Groups with reusable results whose slot went to another program. A slot takes them back when their key and size
	// come again (PrepareBindings).
	struct ParkedGroup {
		std::vector<TextureBinding> members;
		Group                       group;
	};
	std::vector<ParkedGroup> parked;
	size_t                   parked_next = 0;
	// The image heap slots this draw resolves (GuestImageHeap::TakeWork), one entry a dimension the
	// slot's texture can be read as.
	struct HeapImage {
		TextureBinding                        texture;
		ShaderRecompiler::IR::ImageResource   resource;
		ShaderRecompiler::IR::DescriptorValue descriptor;
		uint32_t                              slot = 0;
		uint32_t                              info = 0;
	};
	std::vector<HeapImage>                heap_images;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              program_constants;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
// Whether an image store through `resource` can write `descriptor`; when it cannot, the run ends
// unless `fatal` is false.
bool ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size, bool fatal = true);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
