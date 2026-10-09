#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "speculation-state.h"

#include <array>
#include <atomic>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
struct TextureCacheTestAccess;

class TextureCache {
public:
	// Every image as a tab-separated row (live "images <path>").
	void WriteReport(const char* path);

	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	// Addresses that gained their first registered image, in registration
	// order: a caller that proved "no image starts here" at StartEpoch() E only
	// needs the addresses logged after E.
	[[nodiscard]] uint64_t StartEpoch() const { return m_start_epoch.load(std::memory_order_acquire); }
	[[nodiscard]] bool     HasImageStartingAt(uint64_t address);
	// Appends the (epoch, address) entries logged after `epoch`; false when the
	// log no longer reaches back that far.
	[[nodiscard]] bool NewImageStartsSince(uint64_t epoch, std::vector<std::pair<uint64_t, uint64_t>>& out);
	// The image (of `serial`) still registered, and no image registered or unregistered over its pages since `epoch`
	// (a speculation's work used it: speculation.cpp).
	[[nodiscard]] bool StillResolved(const Image& image, uint64_t serial, uint64_t epoch) const;
	[[nodiscard]] uint64_t ResolutionEpoch() const {
		return m_resolution_epoch.load(std::memory_order_acquire);
	}
	// Only reuses discovery. FindTexture still refreshes contents and selects the current view.
	[[nodiscard]] bool TryReuseSampledImage(ImageId id, const ImageDesc& desc, uint64_t& epoch);
	// TryReuseSampledImage without the resolution-epoch proof, for caches that must
	// survive image registrations elsewhere (native XPR records): the owner is
	// still registered, not being rebound, and has the same backing and resources.
	// Also for a storage binding (an image a record's programs only write); a sampled image may have
	// DCC metadata (whose pending fast clears FindTexture applies: the caller checks IsMetaCleared), and with
	// `storage_dcc` one written as storage too.
	[[nodiscard]] bool IsSampledImageCurrent(ImageId id, const ImageDesc& desc, bool storage_dcc = false);
	// A depth image sampled through a view of its memory (FindImage takes it through ResolveDepthOverlap, a 32-bit
	// float view of a D32S8 image having no format of the same class, or through its stencil plane's association): the
	// lookup finds it again while no image was registered or unregistered over the view's pages since `epoch` (the same
	// owners, whose descriptions do not change), and its use needs nothing of FindTexture but the LRU while no CPU,
	// GPU-buffer or stencil-plane write is pending.
	[[nodiscard]] bool IsSampledDepthCurrent(ImageId id, const ImageDesc& desc, uint64_t epoch);
	// Whether a render target lookup of `requested` (FindImage, any format) still finds `id`: nothing
	// registered over its first page since `epoch`, or no newer image with the requested backing there
	// (it would take precedence). `epoch` moves to the current one when it holds.
	[[nodiscard]] bool IsTargetCurrent(ImageId id, const ImageInfo& requested, uint64_t& epoch);
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
	// A speculative translation's use of the image, when it is committed (`serial`: the image it used).
	void TouchSpeculated(Image& image, uint64_t serial);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size, const char* source = "");
	[[nodiscard]] bool HasTrackedDataOverlap(uint64_t address, uint64_t size);
	// An image over the range holds data the GPU wrote to it (a download of the range must come
	// from the image, not from the buffer).
	[[nodiscard]] bool HasGpuWrittenImageOverlap(uint64_t address, uint64_t size);
	enum class ReadOnlyBufferOverlap { None, CpuSampled, Unsafe };
	[[nodiscard]] ReadOnlyBufferOverlap ClassifyReadOnlyBufferOverlap(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	// Bumped on every insertion, erasure or type change of a surface-metadata
	// record, so a cached negative IsMeta/ClearMeta answer for an address stays
	// valid while this value is unchanged.
	[[nodiscard]] uint64_t MetaEpoch() const {
		return m_meta_epoch.load(std::memory_order_acquire);
	}
	// Moves with any image's barrier state or GPU-written state, registration, metadata or CPU-written bytes: images
	// found in place (a table draw's targets and set) stay so while it does not.
	[[nodiscard]] uint64_t ImagesEpoch() const {
		return ImageStateEpoch() + m_resolution_epoch.load(std::memory_order_acquire) +
		       m_meta_epoch.load(std::memory_order_acquire) + Image::CpuDirtyEpoch();
	}
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice,
	                                 uint32_t* fill_value = nullptr, bool* fill_known = nullptr);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	// Record deferred DCC state while the original guest dispatch writes the metadata.
	void               TrackDccFill(uint64_t address, uint64_t size, uint32_t fill_value);
	[[nodiscard]] bool ClearMeta(uint64_t address, uint32_t fill_value);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void MapMemory(uint64_t address, uint64_t size);
	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	void RunGarbageCollector();
	// The latest tick of a GPU-written image's download that will write guest memory overlapping
	// the range once the GPU completes it (on the priority thread); 0 when none is pending.
	[[nodiscard]] uint64_t PendingDownloadTick(uint64_t address, uint64_t size);
	// Once per guest flip (the unit of NumFramesBeforeRemoval).
	void AdvanceFrame() noexcept { m_frame.fetch_add(1, std::memory_order_relaxed); }
	[[nodiscard]] uint64_t CurrentFrame() const noexcept { return m_frame.load(std::memory_order_relaxed); }

private:
	enum class TransferDirection { Upload, Download };
	struct TextureTransferPlan;
	struct DownloadPlan;

	struct MetaDataInfo {
		// A guest metadata-fill dispatch may initialize DCC before its render target is bound.
		// PendingDcc retains that exact fill until an image binding classifies the address,
		// without exposing an unconfirmed buffer address to the normal metadata heuristics.
		// Keep all surface metadata in one entry so CMask/FMask can be
		// registered beside HTile and DCC without introducing parallel tracking paths.
		enum class Type : uint8_t { PendingDcc, CMask, FMask, HTile, Dcc };

		Type     type       = Type::PendingDcc;
		uint32_t clear_mask = 0;
		uint32_t fill_value = 0xffffffffu;
		uint64_t fill_size  = 0;
		bool     fill_known = false;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 40, 10>;
	// Resolution epoch of the last (un)registration per image page.
	using ImageEpochTable = MultiLevelPageTable<uint64_t, 20, 40, 10>;

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	// Deleted images a speculation's packet may still read (Spec::PacketsPassed), destroyed once it passed.
	std::vector<std::pair<ImageId, Spec::PacketMarks>> m_retired;
	void                                               EraseRetired();
	void                      FreeImage(ImageId id, const char* site = "");
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id, const char* why = "");
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	// Partial CPU writes of large images (KYTY_PARTIAL_IMAGE_DIRTY). Caller holds m_lock.
	[[nodiscard]] static bool PartialDirtyCandidate(const Image& image);
	// `granule`: the released span grows to these boundaries (counted from the image start).
	[[nodiscard]] bool TryInvalidatePartial(Image& image, uint64_t address, uint64_t size,
	                                        uint64_t granule);
	void               UntrackImagePages(Image& image, uint64_t begin, uint64_t end);
	void               RetrackHoles(Image& image);
	// The holes' pages inside [begin, end) only (page-aligned); the others stay released.
	void               RetrackHoles(Image& image, uint64_t begin, uint64_t end);
	void               FinishRefresh(Image& image);
	[[nodiscard]] bool UploadImagePartial(Image& image);
	// The dirty layers in [first_layer, last_layer) only.
	[[nodiscard]] bool UploadDepthPartial(Image& image, uint32_t first_layer = 0,
	                                      uint32_t last_layer = UINT32_MAX);
	// A depth target binding's refresh of the layers its view covers; false: RefreshImage's.
	[[nodiscard]] bool RefreshDepthLayers(ImageId id, const ImageViewInfo& view);
	void               UpdatePartialHashes(Image& image, bool all);
	[[nodiscard]] bool CheckPartialHashes(const Image& image);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);
	[[nodiscard]] bool               SafeToDownload(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        PrepareDccClear(ImageId id, const ImageDesc& desc);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          RefillUpload(Image& image);
	[[nodiscard]] TextureTransferPlan
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] DownloadPlan BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void UploadStencil(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImageData(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, DownloadPlan plan);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	// `keep_partial`: the image stays dirty in the ranges a scoped refresh left (RefreshDepthLayers).
	void CommitGpuWrite(Image& image, bool keep_partial = false);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, const vk::ImageSubresourceRange& range,
	                const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	void               AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	[[nodiscard]] bool TryDownloadImage(ImageId id);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	// Images with a depth_id (stencil associations): deleting an image looks for the ones
	// pointing at it here instead of walking every slot.
	std::vector<ImageId>                              m_stencil_associations;
	ImagePageTable                                    m_image_page_table;
	// Start address -> number of registered images starting there, and the
	// log of addresses that gained their first one.
	std::unordered_map<uint64_t, uint32_t>            m_image_starts;
	// Start addresses per bucket of 4 KiB pages (hashed), and a bit per bucket that has any (8 KiB: it stays in the
	// cache): a lookup of an address in an empty bucket needs no walk of the map (HasImageStartingAt: every formatted
	// buffer of a table draw or dispatch asks, mostly at per-frame ring addresses no image starts at; the map's node
	// walk was ~1% of the GPU thread on the 1-1 walk).
	static constexpr uint32_t StartBucketBits = 16;
	[[nodiscard]] static size_t StartBucket(uint64_t address) {
		return static_cast<size_t>(((address >> 12u) * 0x9e3779b97f4a7c15ull) >> (64u - StartBucketBits));
	}
	[[nodiscard]] bool MayStartAt(uint64_t address) const {
		const auto bucket = StartBucket(address);
		return ((m_start_bucket_bits[bucket / 64u] >> (bucket % 64u)) & 1u) != 0;
	}
	std::unique_ptr<uint32_t[]>                         m_start_buckets = std::make_unique<uint32_t[]>(size_t {1} << StartBucketBits);
	std::array<uint64_t, (size_t {1} << StartBucketBits) / 64> m_start_bucket_bits {};
	std::atomic<uint64_t>                             m_start_epoch {1};
	std::vector<std::pair<uint64_t, uint64_t>>        m_start_log; // (epoch, address)
	std::atomic<uint64_t>                             m_resolution_epoch {1};
	// Null textures: a 1x1 image per format and type (1D, 2D or 3D).
	std::map<std::pair<vk::Format, Prospero::ImageType>, ImageId> m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	// UploadImagePartial: the transfer plan of each large image it uploads (by Image::serial);
	// a 256-layer mip-mapped array has 2816 subresources to plan otherwise on every refresh.
	struct PartialPlan {
		BindingType                          binding {};
		std::shared_ptr<TextureTransferPlan> plan;
	};
	std::unordered_map<uint64_t, PartialPlan> m_partial_plans;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	std::atomic<uint64_t>                             m_meta_epoch {1};
	// The m_surface_metas keys as bits of a 1024-bit filter: a clear bit proves an address holds
	// no metadata without m_lock (IsMeta and ClearMeta run for the buffers of every dispatch).
	// Inserts set their bit under m_lock; UnmapMemory rebuilds it from the remaining keys (a
	// word only loses bits of erased keys); other erases leave a stale bit (a locked lookup).
	std::array<std::atomic<uint64_t>, 16> m_meta_filter {};
	static uint32_t MetaFilterBit(uint64_t address) {
		return static_cast<uint32_t>((address * 0x9e3779b97f4a7c15ull) >> 54u);
	}
	void MetaFilterAdd(uint64_t address) {
		const auto bit = MetaFilterBit(address);
		m_meta_filter[bit >> 6u].fetch_or(uint64_t {1} << (bit & 63u), std::memory_order_release);
	}
	[[nodiscard]] bool MetaFilterMayHold(uint64_t address) const {
		const auto bit = MetaFilterBit(address);
		return ((m_meta_filter[bit >> 6u].load(std::memory_order_acquire) >> (bit & 63u)) & 1u) != 0;
	}
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_logged_video_memory = 0; // RunGarbageCollector's log line
	uint64_t         m_gc_tick                = 0;
	std::atomic<uint64_t> m_frame {0};
	struct PendingDownload {
		uint64_t address = 0, size = 0, tick = 0;
	};
	std::mutex                   m_download_mutex; // the priority thread removes finished ones
	std::vector<PendingDownload> m_pending_downloads;
	mutable uint32_t m_image_query_epoch      = 0;
	// KYTY_TEXTURE_RESOLVE_PAGES: a resolution proven at an epoch still holds while no
	// image over its range registered or unregistered since.
	ImageEpochTable    m_registration_pages;
	void               StampRegistrationPages(const Image& image, uint64_t epoch);
	[[nodiscard]] bool RegistrationsSince(uint64_t address, uint64_t size, uint64_t epoch) const;
	// FindTexture's PrepareDccClear with this view's metadata would change only the fields a color target binding
	// fills (its clear word: no reader but the comparison that moves the meta epoch): the image's DCC surface is the
	// view's and the metadata record there is DCC's (an unknown or pending one it would make DCC). A pending fast clear
	// is the caller's check.
	[[nodiscard]] bool SameDccSurface(const Image& image, const ImageMetadataInfo& metadata) const;
	// KYTY_IMAGE_GRANULES: a bit per 64 KiB granule that a registered image covered
	// since the last rebuild (a superset); region queries skip the page walk on a miss.
	// Allocated with the cache (never moved): MayHaveImagesRead reads it without m_lock.
	mutable std::vector<uint64_t> m_image_granules;
	mutable uint32_t              m_image_granule_releases = 0;
	void                          MarkImageGranules(uint64_t address, uint64_t size) const;
	[[nodiscard]] bool            MayHaveImages(uint64_t address, uint64_t size) const;
	// MayHaveImages without the rebuild (true while one is due): only reads what holders of m_lock write, so
	// m_lock.ReadShared can ask it.
	[[nodiscard]] bool            MayHaveImagesRead(uint64_t address, uint64_t size) const;
	bool             m_readback_linear_images = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
