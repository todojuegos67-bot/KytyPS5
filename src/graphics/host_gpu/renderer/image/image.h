#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <algorithm>
#include <atomic>
#include <compare>
#include <immintrin.h>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
struct ImageTestAccess;
namespace Spec {
struct ImageEntry;
} // namespace Spec

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool texture       = false;
	bool storage       = false;
	bool render_target = false;
	bool depth_target  = false;
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
	vk::ImageAspectFlags pixel_sampled_aspects;
	vk::ImageAspectFlags other_sampled_aspects;
};

// Scopes a group of transitions that all precede the same recorded command. No
// command executes between transitions in one group, so repeating an identical
// transition of the same image inside a group cannot be covering a hazard.
// Transitions recorded outside any group are never treated as repeats, which is
// what keeps the copy and upload paths correct.
void BeginTransitGroup() noexcept;
void EndTransitGroup() noexcept;

// Moves whenever an image's barrier state (layout, access) is set outside a speculation's own view: what was found
// in place holds while it does not.
[[nodiscard]] uint64_t ImageStateEpoch() noexcept;
void                   MoveImageStateEpoch() noexcept;

class TransitGroup final {
public:
	TransitGroup() noexcept { BeginTransitGroup(); }
	~TransitGroup() { EndTransitGroup(); }
	KYTY_CLASS_NO_COPY(TransitGroup);
};

class Image final {
public:
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] const Barriers& GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	// The whole-image barrier state the calling thread's next transition starts from (a speculative translation's
	// own once it used the image: speculation-state.h).
	[[nodiscard]] const VulkanImageState& CurrentState() const;
	// A speculative translation's work uses the image as it is now (its commit puts it there first: a use without a
	// transition needs this too); null outside one.
	Spec::ImageEntry* SpeculativeEntry();
	// The barrier state set to `state` (each subresource's: `subresources`, unless empty), with the barriers there
	// from the image's (a speculative translation's work is committed where it found the image).
	void RestoreState(const VulkanImageState& state, const std::vector<VulkanImageState>& subresources,
	                  vk::CommandBuffer command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	// Moves whenever an image becomes (maybe) CPU-dirty: a proof that bound images were clean
	// holds while it stays (native XPR records used again within a frame).
	[[nodiscard]] static uint64_t CpuDirtyEpoch() noexcept { return s_cpu_dirty_epoch.load(std::memory_order_acquire); }

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(info.data.address, info.data.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
			DropPartialDirty();
			NoteCpuDirty();
		} else if (ImagePageRangesOverlap(info.data.address, info.data.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
			NoteCpuDirty();
		}
	}

	// Partial CPU writes (KYTY_PARTIAL_IMAGE_DIRTY): while the image is dirty only in the
	// recorded guest ranges, it keeps watching its other pages, so a refresh uploads just the
	// subresources over those ranges. A clean image or one already dirty this way can take one.
	[[nodiscard]] bool CanTakePartialDirty() const noexcept {
		return !m_maybe_cpu_dirty && (!m_cpu_dirty || m_partial_dirty);
	}
	void InvalidateCpuWritePartial(uint64_t begin, uint64_t end) {
		if (!CanTakePartialDirty() || begin >= end) {
			EXIT("image cannot take a partial CPU write\n");
		}
		if (!m_cpu_dirty) {
			m_cpu_dirty        = true;
			m_partial_dirty    = true;
			m_maybe_hash_valid = false;
			m_dirty_ranges.clear();
		}
		AddRange(m_dirty_ranges, begin, end);
		NoteCpuDirty();
	}
	[[nodiscard]] bool IsPartiallyCpuDirty() const noexcept { return m_cpu_dirty && m_partial_dirty; }
	[[nodiscard]] const std::vector<std::pair<uint64_t, uint64_t>>& CpuDirtyRanges() const noexcept {
		return m_dirty_ranges;
	}
	// The whole image must be uploaded again (the image stopped watching all of its pages).
	void DropPartialDirty() noexcept {
		m_partial_dirty = false;
		m_dirty_ranges.clear();
	}
	// A refresh of [begin, end) only (the layers one binding covers): the other ranges stay dirty.
	void RefreshRangeComplete(uint64_t begin, uint64_t end) {
		if (!IsPartiallyCpuDirty() || m_maybe_cpu_dirty || begin >= end) {
			EXIT("image cannot complete a partial refresh\n");
		}
		SubtractRange(m_dirty_ranges, begin, end);
		if (m_dirty_ranges.empty()) {
			RefreshComplete();
		}
	}
	// Removes [begin, end) from sorted, disjoint ranges.
	static void SubtractRange(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t begin,
	                          uint64_t end) {
		std::vector<std::pair<uint64_t, uint64_t>> kept;
		kept.reserve(ranges.size() + 1);
		for (const auto& [range_begin, range_end]: ranges) {
			if (range_end <= begin || range_begin >= end) {
				kept.emplace_back(range_begin, range_end);
				continue;
			}
			if (range_begin < begin) kept.emplace_back(range_begin, begin);
			if (end < range_end) kept.emplace_back(end, range_end);
		}
		ranges = std::move(kept);
	}
	// Sorted, disjoint [begin, end) ranges; merges adjacent and overlapping ones.
	static void AddRange(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t begin,
	                     uint64_t end) {
		auto at = ranges.begin();
		while (at != ranges.end() && at->second < begin) ++at;
		auto last = at;
		while (last != ranges.end() && last->first <= end) {
			begin = std::min(begin, last->first);
			end   = std::max(end, last->second);
			++last;
		}
		at = ranges.erase(at, last);
		ranges.insert(at, {begin, end});
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
			NoteCpuDirty();
		}
	}
	[[nodiscard]] bool NeedsMaybeCpuHash() const {
		return m_maybe_cpu_dirty && !m_maybe_hash_valid;
	}
	void SetMaybeCpuHash(uint64_t hash) {
		if (!NeedsMaybeCpuHash()) {
			EXIT("image cannot initialize maybe-dirty hash\n");
		}
		m_maybe_cpu_hash   = hash;
		m_maybe_hash_valid = true;
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || !m_maybe_hash_valid || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		DropPartialDirty();
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept { m_gpu_modified = true; }
	void               ClearGpuModified() noexcept {
		m_gpu_modified = false;
		MoveImageStateEpoch();
	}

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept {
		m_buffer_modified = true;
		MoveImageStateEpoch();
	}
	void               ClearBufferModified() noexcept { m_buffer_modified = false; }

	[[nodiscard]] bool IsStencilModified() const noexcept { return m_stencil_modified; }
	void               MarkStencilModified() noexcept {
		m_stencil_modified = true;
		MoveImageStateEpoch();
	}
	void               ClearStencilModified() noexcept { m_stencil_modified = false; }

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
		return backing.image == nullptr ? 0 : (info.data.size + 1023) & ~uint64_t {1023};
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;

	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	// Page-aligned ranges inside the tracked pages the image stopped watching after partial
	// CPU writes; always inside CpuDirtyRanges while the image is partially dirty.
	std::vector<std::pair<uint64_t, uint64_t>> untracked_holes;
	// KYTY_PARTIAL_IMAGE_DIRTY=2: guest data hashes per partial-dirty granule at the last upload.
	std::vector<uint64_t> partial_hashes;
	ImageId          depth_id {};
	uint64_t         frame_accessed_last = 0; // TextureCache::AdvanceFrame count at the last use
	size_t           lru_id             = 0;
	uint64_t         lru_tick           = 0; // the collection tick of its last use (TextureCache)
	// Transit group that last set the whole-image state; see BeginTransitGroup.
	uint64_t         transit_group      = 0;
	// Unique per image object: a deleted image's slot id goes to later images.
	uint64_t         serial             = 0;
	// The staging copy of the last whole-image upload (null: none or not refillable) and the command buffer it was
	// recorded in (CommandScheduler::CommandSerial): TextureCache::InitializeImage.
	const Buffer*    staged_ring        = nullptr;
	uint64_t         staged_offset      = 0;
	uint64_t         staged_serial      = 0;
	// The barrier state (backing.state, backing.subresource_states, transit_group): changed under this lock, which a
	// speculative translation's thread reads it under (SpeculativeEntry).
	struct StateLock {
		explicit StateLock(const Image& owner) noexcept: image(owner) {
			while (image.state_busy.exchange(true, std::memory_order_acquire)) _mm_pause();
		}
		~StateLock() { image.state_busy.store(false, std::memory_order_release); }
		KYTY_CLASS_NO_COPY(StateLock);
		const Image& image;
	};
	mutable std::atomic<bool> state_busy {false};
	// RegisterImage calls on this object: a proof names one registration.
	uint32_t         registrations      = 0;

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	uint64_t          m_maybe_cpu_hash   = 0;
	static void NoteCpuDirty() noexcept { s_cpu_dirty_epoch.fetch_add(1, std::memory_order_release); }
	inline static std::atomic<uint64_t> s_cpu_dirty_epoch {1};
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	bool              m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
	bool              m_stencil_modified  = false;
	bool              m_partial_dirty     = false;
	std::vector<std::pair<uint64_t, uint64_t>> m_dirty_ranges;
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
