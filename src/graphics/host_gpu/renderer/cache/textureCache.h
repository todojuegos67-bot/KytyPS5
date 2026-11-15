#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/smallVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/cmdRecorder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <map>
#include <span>
#include <type_traits>
#include <mutex>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class TextureCache;
struct TextureCacheTestAccess;

namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
} // namespace BcAlias

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
		bool          heap = false; // PS5: resolved for a resident image heap.
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	// With `create` false only an image that exists is returned (or a view of one): none is made.
	// `plain` (when given) is the found image's lookup stamp when the answer depended on nothing but the images over
	// the request's pages: an image of the same backing was found (not a stencil plane's record), none of the
	// candidates is held by level or a block-compressed alias of the request, no grow was pending, nothing was made,
	// freed or viewed at another level, and `desc` is unchanged. 0 otherwise. While that image's stamp stands, the
	// same lookup finds it again, and RepeatLookup does the rest of what it did.
	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false, bool create = true,
	                                      uint64_t* plain = nullptr);
	// What a plain lookup that found `id` with `stamp` did beside finding it (the access tick, the LRU touch, the
	// color-clear materialization), for the same `desc` and its view's base layer as it was asked. False, with nothing
	// done, when the image's stamp moved or a grow is pending (FindImage completes those first): look up again.
	[[nodiscard]] bool          RepeatLookup(ImageId id, uint64_t stamp, const ImageDesc& desc,
	                                         uint32_t metadata_base_layer);
	// The image's lookup stamp (FindImage), 0 for none.
	[[nodiscard]] uint64_t      LookupStamp(ImageId id);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);
	// An image the GPU writes (a render target, a storage image) holds this address.
	[[nodiscard]] bool HasGpuImageAt(uint64_t address);
	[[nodiscard]] bool HasAnyImageAt(uint64_t address);
	// The image heap shows an image based at this address, other than `own`.
	[[nodiscard]] bool HasHeapImageAt(uint64_t address, ImageId own);
	// A target of the draw being prepared, or its stencil plane, overlaps these bytes.
	[[nodiscard]] bool HasTargetIn(uint64_t address, uint64_t size);
	// The bytes HasTargetIn looks for when these images are the draw's targets: theirs and their linked stencil planes',
	// and those of an image a target merged into, which keeps the flag (ResolveOverlap). No other image can make it true.
	void AppendTargetBytes(std::span<const ImageId> targets, std::vector<std::pair<uint64_t, uint64_t>>& bytes);
	// What RepeatLookup reads besides an image's stamp. While it stands every stamp stands too (RestampOwners), so a
	// lookup repeated under it repeats again, and its touch has nothing left to do (LruCache::Touch).
	struct LookupState {
		uint64_t clock = UINT64_MAX;
		uint64_t tick  = UINT64_MAX;
		bool     grows = true;
		bool     operator==(const LookupState&) const = default;
	};
	[[nodiscard]] LookupState GetLookupState() {
		std::scoped_lock lock {m_lock};
		return {m_image_change_clock, m_gc_tick, !m_pending_grows.empty()};
	}
	// The one part of RepeatLookup that reads guest memory, so nothing can stand in for it: a color image's metadata can
	// ask for a clear at any time. Such a check can change the state above.
	[[nodiscard]] static bool HasColorClearCheck(const ImageDesc& desc) {
		return !desc.info.data.Empty() && (desc.info.metadata.kind == ImageMetadataKind::Dcc ||
		                                   desc.info.metadata.kind == ImageMetadataKind::Cmask);
	}
	void RepeatColorClear(ImageId id, const ImageDesc& desc, uint32_t metadata_base_layer) {
		MaterializeColorClear(id, desc, metadata_base_layer);
	}
	// An image of this backing exists: a lookup for it neither makes nor replaces one.
	[[nodiscard]] bool HasImageFor(const ImageInfo& info);
	// Counts changes to the set of images: registrations, unregistrations and images marked for rebind. While it
	// stands, a lookup that found no image, or HasImageFor, gives the same answer (group reuse and the texture memo,
	// PrepareBindings).
	[[nodiscard]] uint64_t ImageSetEpoch() const { return m_image_set_epoch.load(std::memory_order_relaxed); }
	// The changes RestampOwners made (images registered or unregistered, stencil records linked again), each as the
	// bytes of the pages it covered: HasImageFor, or a lookup that found no image, gives the same answer for bytes no
	// later change covers (group reuse, PrepareBindings). ImageChangesSince calls visit(address, end) for every change
	// after `clock`, false when the list no longer holds them all.
	[[nodiscard]] uint64_t ImageChangeClock() {
		std::scoped_lock lock {m_lock};
		return m_image_change_clock;
	}
	template <typename Visit>
	[[nodiscard]] bool ImageChangesSince(uint64_t clock, Visit&& visit) {
		std::scoped_lock lock {m_lock};
		if (m_image_change_clock - clock > m_image_changes.size()) return false;
		for (; clock < m_image_change_clock; ++clock) {
			const auto& [address, end] = m_image_changes[clock % m_image_changes.size()];
			visit(address, end);
		}
		return true;
	}

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	// Writes a copy into whole levels of one layer of a GPU-written image straight into it, and
	// returns false for any other copy.
	[[nodiscard]] bool CopyToImage(uint64_t address, uint64_t source, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);
	// An image or surface metadata in the range: what UnmapMemory would free.
	[[nodiscard]] bool HoldsRegion(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	// With storage readback, a GL2 writeback makes GPU-written images visible to the CPU; returns
	// whether a download was recorded.
	[[nodiscard]] bool WriteBackImages();
	// The tick of the latest write-back pending over the range's pages, or 0 when none is.
	[[nodiscard]] uint64_t WriteBackTick(uint64_t address, uint64_t size);
	void RunGarbageCollector();

	// A streamed texture is held by level (sparse residency, HeldLevels) when its levels can get
	// memory one by one: a block-compressed tiled 2D chain whose finest level fills a sparse block.
	// It is one image for life, and a lower min LOD binds and uploads only the levels added.
	[[nodiscard]] bool HoldsLevels(Prospero::BufferFormat format, Prospero::TileMode tile, uint32_t width,
	                               uint32_t height, bool metadata, uint32_t first);
	// Before an unmap: the game drops a streamed texture's finest levels by unmapping their pages,
	// after it raised its min LOD. An image held by level gives those levels' memory back and keeps
	// the rest, so the unmap does not have to wait for the GPU.
	void ReleaseLevels(uint64_t address, uint64_t size);

private:
	struct MetaDataInfo {
		// PS5 DB_DEPTH_VIEW encodes 13-bit base and last array-slice indices.
		std::bitset<8192> clear_mask;
		// The depth image that last acquired this HTile range.
		ImageId owner;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	// A recorded image download whose guest pages stay inaccessible until it lands.
	struct WriteBack {
		GuestRange range;
		uint64_t   tick;
	};

	// PS5: guest addresses span 44 bits, so an entry keeps the whole image address.
	struct BucketEntry {
		uint64_t address;
		uint32_t size;
		ImageId  id;

		[[nodiscard]] bool Overlaps(uint64_t addr, size_t size) const noexcept {
			return address < addr + size && addr < address + this->size;
		}
	};

	struct alignas(64) Bucket {
		Common::SmallVector<BucketEntry, 3, uint32_t> entries;
	};
	static_assert(sizeof(Bucket) == 64);

	using ImagePageTable = MultiLevelPageTable<Bucket, 20, 44, 14>;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		static constexpr bool RETURNS_BOOL =
		    std::is_same_v<std::invoke_result_t<Func, uint64_t>, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	// The walks below iterate the page entries in place: func must not insert, register,
	// unregister or free images. Caller holds m_lock.

	// Calls func once per image overlapping the region, from the first page both share; a func
	// returning true stops the walk.
	template <typename Func>
	void ForEachImageInRegion(uint64_t address, size_t size, Func&& func) {
		static constexpr bool BOOL_BREAK =
		    std::is_same_v<std::invoke_result_t<Func, ImageId, Image&>, bool>;
		ImagePageTable::PageRange pages {};
		if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
			return;
		}
		const uint64_t first_page = address >> ImagePageTable::kPageBits;
		ForEachPage(address, size, [&](uint64_t page) {
			const auto* bucket = m_image_page_table.Find(page);
			if (bucket == nullptr) {
				if constexpr (BOOL_BREAK) {
					return false;
				} else {
					return;
				}
			}
			for (const auto& entry: bucket->entries) {
				const uint64_t base_page = entry.address >> ImagePageTable::kPageBits;
				if (page != std::max(first_page, base_page)) {
					continue;
				}
				if (!entry.Overlaps(address, size)) {
					continue;
				}
				Image& image = m_slot_images[entry.id];
				if constexpr (BOOL_BREAK) {
					if (func(entry.id, image)) {
						return true;
					}
				} else {
					func(entry.id, image);
				}
			}
			if constexpr (BOOL_BREAK) {
				return false;
			}
		});
	}

	// Calls func for each image registered at address.
	template <typename Func>
	void ForEachImageWithAddress(uint64_t address, Func&& func) {
		const auto* bucket = m_image_page_table.Find(address >> ImagePageTable::kPageBits);
		if (bucket == nullptr) {
			return;
		}
		for (const auto& entry: bucket->entries) {
			if (entry.address == address) {
				func(entry.id, m_slot_images[entry.id]);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info, uint32_t resident = HeldLevels::NotHeld);
	// An image held by level: memory and texels from level `first` on.
	void                      HoldLevels(ImageId id, uint32_t first);
	// Deferred grows whose bind the bind thread has sent: the images show the new levels.
	void                      CompleteGrows();
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	// A new lookup stamp for every image over these pages (Image::lookup_stamp).
	void                      RestampOwners(uint64_t address, uint64_t size);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	// Caller holds m_lock and ends rendering. Begins rendering `layers` layers of one color mip
	// through a `format` view, clearing them or, when `clear` is null, loading them; returns the
	// render area.
	vk::Rect2D BeginColorRendering(Image& image, vk::Format format, uint32_t level,
	                               uint32_t layer, uint32_t layers, const vk::ClearValue* clear,
	                               Cmd command);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	// The depth image's stencil plane record covering exactly range, or none.
	[[nodiscard]] ImageId FindStencilPlane(const Image& depth, GuestRange range) const;
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	[[nodiscard]] bool DownloadQueuedImages();

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	std::map<std::pair<vk::Format, Prospero::ImageType>, ImageId> m_null_images; // by format and base type
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	GroupNotices                                      m_group_notices;
	std::mutex                                        m_write_back_mutex;
	std::vector<WriteBack>                            m_write_backs;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	std::atomic<uint64_t> m_image_set_epoch {0}; // ImageSetEpoch
	uint64_t         m_lookup_clock           = 0; // the last lookup stamp given (RestampOwners)
	std::array<std::pair<uint64_t, uint64_t>, 4096> m_image_changes {}; // ImageChangesSince, a ring
	uint64_t         m_image_change_clock     = 0;
	bool             m_readback_linear_images = false;
	std::vector<ImageId> m_pending_grows; // held images with a deferred grow (CompleteGrows)
	std::vector<ImageId> m_merged_targets; // images a target merged into, while they keep its flag (AppendTargetBytes)
	bool             m_readback_tiled_images  = false;
	bool             m_readback_storage_images = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class ImageHeap;
	friend class GuestImageHeap;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
