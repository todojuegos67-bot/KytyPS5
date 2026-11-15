#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "graphics/host_gpu/renderer/image/groupNotices.h"
#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/cmdRecorder.h"

#include <array>
#include <atomic>
#include <compare>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
class TileManager;
struct ImageTestAccess;

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool storage       = false;
	bool render_target = false;
	bool video_out     = false;
};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
};

// A streamed texture held by level (TextureCache::HoldsLevels): a sparse image of the whole chain
// with memory for the levels from `resident` on. The image's info.data covers only those levels'
// guest bytes, the start of the texture's range.
struct HeldLevels {
	static constexpr uint32_t NotHeld = UINT32_MAX;

	// How many levels a view of `count` levels from `base` has left once it starts at `resident`;
	// one if all of them are below it.
	[[nodiscard]] static constexpr uint32_t ViewLevels(uint32_t base, uint32_t count, uint32_t resident) {
		return base + count > resident ? base + count - resident : 1u;
	}

	uint32_t                      resident  = 0;  // the first level with memory and texels
	uint32_t                      tail      = 0;  // the first level of the host's mip tail, bound for life
	uint32_t                      upload    = 0;  // levels from `resident` on still to upload; 0: all
	uint64_t                      full_size = 0;  // the whole chain's guest bytes
	uint64_t                      batch     = 0;  // the last sparse batch that binds this image
	uint32_t                      pending   = NotHeld; // a deferred grow to this level, until its bind is sent
	uint64_t                      pending_batch = 0;
	vk::Extent3D                  block {};       // the sparse block, in texels
	vk::MemoryRequirements        memory {};
	std::array<VmaAllocation, 16> levels {};      // by level, below the tail
	std::vector<VmaAllocation>    tails;
};

class Image final {
public:
	// `resident` other than NotHeld: held by level, from that level on.
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info,
	      uint32_t resident = HeldLevels::NotHeld);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	// Held by level: the levels from `first` on get memory, the ones before it lose theirs.
	// `deferred`: only bind the new levels (a grow the image shows once the bind is sent).
	void HoldLevels(uint32_t first, bool deferred = false);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, Cmd command_buffer);
	// The stage Transit gives an access.
	[[nodiscard]] static vk::PipelineStageFlags2 TransitStage(vk::AccessFlags2 destination_access);
	// Every subresource Transit would cover is already in the state it asks for (layout, access and stage): Transit
	// would then change no layout and no tracked state, and only order earlier accesses of the image before later
	// ones, which a memory barrier with the same masks does too.
	[[nodiscard]] bool InState(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	                           std::optional<ImageSubresourceRange> range) const;
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	void MarkCpuDirty() {
		GroupNotice(GroupNotices::Kind::Contents);
		m_cpu_dirty       = true;
		m_maybe_cpu_dirty = false;
		if (held) held->upload = 0;
		NotifyHeap();
		HeapChanged();
	}
	// Held by level: the `count` levels from `held->resident` on are new and need their texels.
	void MarkLevelsDirty(uint32_t count) {
		GroupNotice(GroupNotices::Kind::Contents);
		held->upload      = IsCpuDirty() && held->upload == 0 ? 0u : held->upload + count;
		m_cpu_dirty       = true;
		m_maybe_cpu_dirty = false;
		HeapChanged();
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!IsCpuDirty()) {
			GroupNotice(GroupNotices::Kind::Contents);
			m_maybe_cpu_hash  = HashGuestEdges();
			m_maybe_cpu_dirty = true;
			HeapChanged();
		}
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if ((!m_maybe_cpu_dirty || m_cpu_dirty) && !KYTY_UPSTREAM_PATH("ef8a6d5f6")) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty() && !KYTY_UPSTREAM_PATH("21f3b9de6")) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty       = false;
		m_maybe_cpu_dirty = false;
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept { m_gpu_modified = true; }
	void               ClearGpuModified() noexcept {
		if (m_gpu_modified) GroupNotice(GroupNotices::Kind::Contents);
		m_gpu_modified = false;
	}

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept {
		GroupNotice(GroupNotices::Kind::Contents);
		m_buffer_modified = true;
		HeapChanged();
	}
	void               ClearBufferModified() noexcept { m_buffer_modified = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : Common::AlignUp(info.data.size, 1024);
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;
	[[nodiscard]] vk::ImageLayout ReadLayout() const noexcept {
		return info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
		                      : vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	// Bound through a storage group: its changes are listed from here on (GroupNotices).
	void EnrollGroupNotices(GroupNotices& notices, uint32_t id) noexcept {
		group_notices = &notices;
		group_image   = id;
	}
	void GroupNotice(GroupNotices::Kind kind) const noexcept {
		if (group_notices != nullptr) group_notices->Add(group_image, kind);
	}

	// Resident image heaps sample an image without rediscovering it until one changes its
	// contents, tracking, identity or layout.
	void HeapChanged() const noexcept {
		if (heap_refs.load(std::memory_order_relaxed) != 0u) {
			heap_epoch.fetch_add(1u, std::memory_order_relaxed);
		}
	}

	static inline std::atomic<uint64_t> heap_epoch {0};
	static inline std::atomic<uint64_t> serials {0};

	// Unique per image, unlike ImageId: freed slots are reused.
	const uint64_t   serial = serials.fetch_add(1u, std::memory_order_relaxed) + 1u;
	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	GroupNotices*    group_notices = nullptr;
	uint32_t         group_image   = 0;
	bool             registered     = false;
	// New whenever an image is registered or unregistered over this one's pages, or this stencil record is linked
	// again (TextureCache::RestampOwners): a plain lookup that found this image finds it again while it stands.
	uint64_t         lookup_stamp   = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	ImageId          depth_id {};
	// Stencil associations that refer to this depth image.
	std::vector<ImageId> stencil_ids;
	// The current stencil plane's mapping into the depth image, also retained by its association.
	ImageSubresourceRange stencil_subresources;
	uint64_t         tick_accessed_last = 0;
	size_t           lru_id             = 0;
	// The tick of its last touch of the LRU list, so that a repeated touch needn't read the list's entry.
	uint64_t         lru_tick           = 0;
	// Set once the guest image heap shows the image: its layout and texel changes go to the heap.
	ImageId          heap_id {};
	std::unique_ptr<HeldLevels> held; // null unless held by level
	// Resident image heap elements that select this image.
	std::atomic<uint32_t> heap_refs {0};
	// Set once a kept group set holds its views (RenderExecutor::CommitBindings); its destruction then moves
	// group_set_epoch, so no kept set is bound again with a view handle that may come back for another image.
	bool                         in_group_set = false;
	inline static std::atomic<uint64_t> group_set_epoch {0};

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	void NotifyHeap();
	[[nodiscard]] VmaAllocation AllocateHeld(uint64_t size, vk::DeviceMemory& memory, vk::DeviceSize& offset);
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	uint64_t          m_maybe_cpu_hash  = 0;
	bool              m_cpu_dirty       = false;
	bool              m_maybe_cpu_dirty = false;
	bool              m_gpu_modified    = false;
	bool              m_buffer_modified = false;
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);
// The sparse block of a 2D texture of this format made as Image makes it, in texels; zero when
// the device cannot bind such an image's levels one by one.
[[nodiscard]] vk::Extent3D           SparseBlock(GraphicContext& graphics, Prospero::BufferFormat format);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
