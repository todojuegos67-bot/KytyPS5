#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <mutex>
#include <compare>
#include <limits>
#include <atomic>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
class TileManager;
class CommandRecorder;
struct ImageTestAccess;

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

// What the image itself keeps of its use by the draw being emitted: the attachment state set
// while recording (AcquireRenderTargets), and whether the cache replaced it under a draw that
// had bound it. Cleared by RenderExecutor::ResetBindings. MT GPU stage C (gt7-mt-1117): the
// binding flags proper (bound, target, storage) moved to ImageBindingTable, per translation
// context; these stay here because only the emission (serialised, scheduler lock) writes the
// attachment state, and needs_rebind is the cache's verdict on the image, whoever bound it.
struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             needs_rebind = false;
};

// MT GPU stage C: one translation context's binding flags of one image (were ImageBinding's).
struct ImageBindFlags {
	bool is_bound      = false;
	bool is_target     = false;
	bool force_general = false;
	bool shader_write  = false;

	ImageBindFlags& operator|=(const ImageBindFlags& o) noexcept {
		is_bound |= o.is_bound;
		is_target |= o.is_target;
		force_general |= o.force_general;
		shader_write |= o.shader_write;
		return *this;
	}
	bool operator==(const ImageBindFlags&) const = default;
};

// MT GPU stage C: the binding flags of the images one translation context has bound (a few
// dozen at most per draw: a flat vector, in the order they were first bound). Written by its
// context (BindImage, BindRenderTarget, ResetBindings) and by the texture cache (merges,
// destruction); read by both. Owned by RenderExecutor::TranslationContext; the texture cache
// reads every registered table (TextureCache::BindFlags). Not thread-safe by itself: written
// and read under the texture cache's lock.
class ImageBindingTable {
public:
	[[nodiscard]] const ImageBindFlags* Find(ImageId id) const noexcept {
		for (const auto& [key, flags]: m_entries) {
			if (key == id) return &flags;
		}
		return nullptr;
	}
	[[nodiscard]] ImageBindFlags Get(ImageId id) const noexcept {
		const auto* flags = Find(id);
		return flags != nullptr ? *flags : ImageBindFlags {};
	}
	ImageBindFlags& At(ImageId id) {
		for (auto& [key, flags]: m_entries) {
			if (key == id) return flags;
		}
		return m_entries.emplace_back(id, ImageBindFlags {}).second;
	}
	void Erase(ImageId id) noexcept {
		for (size_t i = 0; i < m_entries.size(); i++) {
			if (m_entries[i].first == id) {
				m_entries.erase(m_entries.begin() + static_cast<std::ptrdiff_t>(i));
				return;
			}
		}
	}
	void Clear() noexcept { m_entries.clear(); }
	[[nodiscard]] const std::vector<std::pair<ImageId, ImageBindFlags>>& Entries() const noexcept {
		return m_entries;
	}

private:
	std::vector<std::pair<ImageId, ImageBindFlags>> m_entries;
};

// Round 167 (images:lean): while one lives on the GPU thread, the image uses made are a draw's or
// a dispatch's (its render targets and textures, RenderExecutor), which the guest's own sync
// points order. Not an emulator-made copy, blit, clear or conversion.
class GuestImageUses final {
public:
	GuestImageUses() noexcept: m_previous(t_active) { t_active = true; }
	~GuestImageUses() { t_active = m_previous; }
	GuestImageUses(const GuestImageUses&)                      = delete;
	GuestImageUses&           operator=(const GuestImageUses&) = delete;
	[[nodiscard]] static bool Active() noexcept { return t_active; }

private:
	static inline thread_local bool t_active = false;
	bool                            m_previous;
};

class Image final {
public:
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	// Records through the recording thread when given a threaded CommandBuffer::Recorder(); a raw
	// command buffer converts.
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, const CommandRecorder& command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(info.data.address, info.data.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
		} else if (ImagePageRangesOverlap(info.data.address, info.data.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
		}
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
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

	// TextureCache::RefreshByContent: the guest bytes changed (definitely dirty), or they did not
	// (a maybe-dirty mark is dropped).
	void MarkCpuDirty() {
		m_cpu_dirty        = true;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
	}
	void ClearMaybeCpuDirty() {
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept { m_gpu_modified = true; }
	void               ClearGpuModified() noexcept { m_gpu_modified = false; }

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept { m_buffer_modified = true; }
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

	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	// Referenced by a bindless table slot: the collector keeps it, and unregistering it tells the
	// table (TextureCache::on_bindless_unregister) to repoint the slot.
	bool             bindless_pinned = false;
	ImageId          self_id {}; // set by TextureCache::InsertImage
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	ImageId          depth_id {};
	// A depth image's stencil associations (the images whose depth_id is it): deleting it frees
	// them without a scan of the whole cache (upstream's "avoid full-cache scans when deleting
	// stencil associations").
	std::vector<ImageId> stencil_ids;
	// The current stencil plane's mapping into the depth image, also retained by its association.
	ImageSubresourceRange stencil_subresources;
	uint64_t         tick_accessed_last = 0;
	// Presented frame of the last access: overlapping aliases age by frames as well as by ticks
	// (TextureCache::ResolveOverlap, KYTY_IMAGE_ALIAS_AGE).
	uint64_t         frame_accessed_last = 0;
	uint64_t         frame_guest_dropped = 0; // when the guest last dropped it from its heaps
	size_t           lru_id             = 0;
	// The LRU tick the cache last gave it (round 80): a touch in the same tick skips the list's
	// node, a cache miss on most of the draws' textures.
	uint64_t lru_tick = 0;
	// Last GPU writer among live overlapping aliases; cleared when another alias takes the bytes.
	bool             alias_owner        = false;
	// Kept beside an image of another kind at its address, neither written by the GPU
	// (TextureCache::KeepReadOnlyAlias, round 79): a GPU write to one of them frees the others.
	bool read_only_alias = false;
	// CommandScheduler tick, and metadata slices, MaterializeDccClear last resolved this image's
	// DCC metadata for. UINT64_MAX means never checked. Lets repeated FindImage lookups within
	// the same still-unsubmitted recording (i.e. many draws to the same bound render target) skip
	// the synchronous GPU readback after the first one -- a real re-clear in a later recording
	// still gets caught because the tick will have advanced by then.
	uint64_t         dcc_clear_checked_tick  = UINT64_MAX;
	uint32_t         dcc_clear_checked_first = 0;
	uint32_t         dcc_clear_checked_count = 0;
	// A small linear image the CPU writes (TextureCache::ChecksGuestContent): the hash of the
	// guest bytes its last upload took, and the guest submission they were last compared in.
	uint64_t content_hash             = 0;
	uint64_t content_check_submission = 0;
	bool     content_hash_valid       = false;
	// TextureCache::InitializeImage has run: it holds contents (an upload, or the GPU's) since.
	bool initialized = false;
	// It holds pixels the GPU wrote (a draw, store, clear or copy) since its last upload.
	bool gpu_pixels = false;
	// Counts the uploads and GPU writes (TextureCache::ContentSerial): diagnostics scan an image
	// again only once it changed.
	uint64_t content_serial = 0;
	// TextureCache::ResetUnmappedDepth: the content it last checked (UINT64_MAX: none yet).
	uint64_t sparse_checked_serial = UINT64_MAX;
	// Round 167 (images:lean): an emulator-made use (a copy, blit, clear, conversion) since its
	// last barrier, which no guest sync point orders: the next use makes its barrier.
	bool internal_use = false;

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
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	bool              m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
};

// Changes of a bindless-pinned image's state that the per-draw bindless pass checks
// (RenderExecutor::CommitBindings): its layout, render-target use and registration. A heap is
// rechecked only when this changed since it was last checked, or for the images resolved since.
inline std::atomic<uint64_t> g_bindless_state_generation {1};
// The pinned images whose state changed, for the engine method's per-draw pass (E5), which looks
// at these and at newly settled images only, instead of every image of a heap.
struct BindlessStateChanges {
	std::mutex           mutex;
	std::vector<ImageId> ids;
};
inline BindlessStateChanges g_bindless_state_changes;
inline void NoteBindlessStateChange(const Image& image) noexcept {
	if (image.bindless_pinned) {
		g_bindless_state_generation.fetch_add(1, std::memory_order_relaxed);
		std::scoped_lock lock {g_bindless_state_changes.mutex};
		g_bindless_state_changes.ids.push_back(image.self_id);
	}
}

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
