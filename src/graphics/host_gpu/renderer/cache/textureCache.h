#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/imageZoneStamps.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/conditionalClear.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
struct TextureCacheTestAccess;

// How TextureCache::ClearImageFromBuffer takes a fill.
struct BufferFillClear {
	// Refuse when another image starts at the fill's address: a fill there reaches it too (a
	// dispatch's buffer write gives it the written bytes), and a clear would not. Stencil planes
	// that no buffer write reaches (multisampled or in an HTile) do not count. With `also_run`,
	// such a plane is cleared anyway and *also_run says the fill must still run for the others.
	bool  alone    = false;
	bool* also_run = nullptr;
	// The word is the guest memory's: a 16-bit depth target's word holds two of its texels.
	bool guest_word = false;
	// What it cleared, or why it did not, for logs (optional).
	std::string* what = nullptr;
};

// The "Image churn" line (round 78: images registered and unregistered, by area, why dropped and
// which most often), when 10 s have passed since the last. Not under the texture cache's lock.
void ReportImageChurnIfDue();

class TextureCache {
public:
	// Called before a bindless-pinned image is unregistered.
	std::function<void(ImageId)> on_bindless_unregister;
	// Called right before an image is destroyed; true keeps it one more GPU tick (a bindless slot
	// still held one of its views and was repointed).
	std::function<bool(const Image&)> on_image_release;
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

	// `same_backing`, when given, says whether the image is one that was there with the same
	// backing (SameBacking), as it was: no overlap had to be resolved and no image was made.
	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false,
	                                      bool* same_backing = nullptr);
	// KYTY_TABLE (tableTargets.h): a render target FindImage found with the same backing before,
	// while ImageSetGeneration has not moved since: what FindImage does besides the lookup (the
	// image's tick and LRU touch, its uint array description, its color clear's
	// materialization). False when the image is gone (the caller then looks it up).
	[[nodiscard]] bool          RefindTarget(ImageId id, const ImageDesc& desc);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	// The texture memo (RenderExecutor::ResolveTexture). The image set's generation changes
	// whenever an image is registered or unregistered: while it stays, FindImage gives a
	// description the same image, so a binding resolved before can name it again.
	[[nodiscard]] uint64_t ImageSetGeneration() const noexcept {
		return m_image_set_generation.load(std::memory_order_acquire);
	}
	// The same for one range of guest memory (round 78): it stays while no image over memory
	// near the range was registered or unregistered (ImageZoneStamps), so FindImage gives a
	// description there the same image. KYTY_BISECT's texmemo:global goes back to the generation.
	[[nodiscard]] uint64_t RangeStamp(uint64_t address, uint64_t size) const noexcept {
		return m_zone_stamps.RangeStamp(address, size);
	}
	// GPU thread: whether a sampled texture binding of image `id` may be kept for reuse at all: a
	// registered image that is not a depth image nor a depth image's stencil plane, without
	// metadata, and not one whose bytes are compared with the guest's (ChecksGuestContent).
	[[nodiscard]] bool MayReuseTextureBinding(ImageId id);
	// GPU thread: whether such a binding of image `id`, and the view FindTexture gave it, can be
	// used again as they were: the image may be kept (above), nothing has written it since (the
	// CPU or a GPU buffer write), and its pages are all tracked, so that FindImage and FindTexture
	// would find it and its view and leave it as it is. It is then touched as FindImage touches it.
	[[nodiscard]] bool ReuseTextureBinding(ImageId id);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	// A fill of exactly an image's memory (a color or depth image's, or a depth image's stencil
	// plane) as a clear of that image to the fill's word; false when there is no such image or the
	// word does not decode for its format.
	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear, const BufferFillClear& how = {});
	// KYTY_PENDING_FILLS (default on, 0 turns it off): a GPU fill that found no image to clear
	// (a compute shader's, which is then noted with BufferCache::NoteFill, or a DMA fill) still
	// holds for its range, the bytes of it that were not mapped then among them: a render or depth
	// target created later inside it starts as the fill instead of what its memory holds then.
	[[nodiscard]] static bool ClearsTargetsFromFills();
	void               InvalidateMemory(uint64_t address, uint64_t size);
	// InvalidateMemory, and whether any image lies in [block, block + block_size) (one lock).
	// image_pages (round 79): the tracker pages of the block an image lies on, bit i for its page
	// i (every bit for a block of more than 64 pages that holds an image).
	[[nodiscard]] bool InvalidateMemoryNear(uint64_t address, uint64_t size, uint64_t block,
	                                        uint64_t block_size, uint64_t* image_pages = nullptr);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	// `fill`: the value a fill of the metadata wrote throughout, when known (a DMA fill, a compute
	// shader's uniform fill); MetaFillValue gives the last one (none for a clear the draw's
	// registers ask for).
	[[nodiscard]] bool ClearMeta(uint64_t address, std::optional<uint32_t> fill = std::nullopt);
	[[nodiscard]] std::optional<uint32_t> MetaFillValue(uint64_t address);
	// The size of the HTile that starts at the address (its depth target's last binding), or 0.
	[[nodiscard]] uint64_t MetaSize(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);
	// Whether a color target's CMASK or DCC starts at this address (MaterializeColorClear notes
	// them): a fill there that is also a depth target's HTile clear must still run.
	[[nodiscard]] bool IsColorMeta(uint64_t address);
	// Any thread: the ranges image write-backs to guest memory still have to land in
	// (DownloadImageMemory), as (address, size), a copy. A fast readback leaves the bytes under
	// them (BufferCache::FastReadback): such a write-back, on the priority thread, would put older
	// bytes over its own.
	[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>> PendingWriteBacks() const;

	void UnmapMemory(uint64_t address, uint64_t size);
	// Whether the GPU's writes to an image like this go back to guest memory after each
	// submission, for the CPU to read: linear images up to KYTY_READBACK_SMALL_IMAGES_KB, or every
	// linear one with --readback-linear-images.
	[[nodiscard]] bool ReadsBackForCpu(const ImageInfo& info) const;
	// Round 124, diagnostics, any thread: the images over [address, address + size), one line each
	// (at most `count`): extent, formats, tiling and whether what the GPU wrote reaches guest
	// memory. For a CPU-side encoder's input (a thumbnail that came out as noise).
	[[nodiscard]] std::vector<std::string> DescribeImagesInRange(uint64_t address, uint64_t size,
	                                                             size_t count);
	void ProcessDownloadImages();
	// Round 121: whether a read-back for the CPU is on its way (enrolled, or recorded and not
	// landed). GPU thread.
	[[nodiscard]] bool ReadbacksOnTheirWay();
	// Round 122: for a texture binding that reads a multisample depth image as single-sample (a
	// T# of a 2D image over a multisample depth target: the console's texture unit reads its
	// first sample), a single-sample depth image holding that sample, copied again whenever the
	// depth image changed. Vulkan cannot sample a multisample image through a single-sample
	// binding: what NVIDIA read there was the samples interleaved, rows of two images, so depth
	// tests in shaders (the light sprites' occlusion in replays) and depth-based effects read the
	// wrong depth. Each layer of an array is copied. Any other image: itself (the binding then
	// gets a null texture, RenderExecutor::FindResolvedTexture). KYTY_MS_DEPTH_SAMPLE0=0: itself
	// (so a null texture: for comparisons only). The source is refreshed first, as FindTexture
	// would refresh it. GPU thread.
	[[nodiscard]] ImageId SingleSampleDepth(ImageId id);
	// Round 124: the same for the stencil of a multisample depth/stencil image read through a
	// single-sample binding (a T# of its stencil plane): an R8_UINT image holding the first
	// sample's stencil, copied again whenever the depth image or its stencil plane changed. Any
	// other image: itself. KYTY_MS_DEPTH_SAMPLE0=0 turns this off too. GPU thread.
	[[nodiscard]] ImageId SingleSampleStencil(ImageId depth_id);
	// Round 124 (Franseven's case): a texture narrower or shorter than the single-sample colour
	// image at its address, with the same guest layout (the same size and tiling: text drawn with
	// a narrower viewport into a wider target, then sampled at the narrower width). A view cannot
	// shrink the extent, so normalized coordinates would reach the image's columns past the
	// texture's: the texture reads a copy of the image's corner of its extent, made again
	// whenever the image changed. The image itself stays as it is (the target keeps all its
	// pixels). Any other request: `id`. GPU thread.
	[[nodiscard]] ImageId SubExtentCopy(ImageId id, const ImageInfo& requested);
	void RunGarbageCollector();
	// The guest no longer refers to this bindless image from any heap entry (it replaced or
	// cleared them): free it once no draw has used it for a few frames (engine method).
	void ReleaseDroppedByGuest(ImageId id);
	// Bindless images no draw sampled during the last usage probe (BindlessTable::
	// CollectUnusedImages). They stay pinned, but under memory pressure the collector may free
	// them like any other image; a key that samples one again asks for it anew. Replaces the
	// previous set.
	void SetBindlessEvictable(std::span<const ImageId> ids);
	// Whether the cache's images are above the collector's pressure threshold.
	// All images, pinned ones included, against the bound: eager settling pauses on it. Settling
	// ahead while the collector freed other images under the bound lost the device at ~57 s into
	// a new game, every time (2026-10-07, ISSUES #17); the collector's own thresholds use
	// CollectableMemory.
	[[nodiscard]] bool UnderPressure() const noexcept {
		return m_total_used_memory >= m_pressure_gc_memory;
	}
	// Image bytes the collector may free: all but the bindless-pinned ones (refreshed once a frame).
	[[nodiscard]] uint64_t CollectableMemory() const noexcept {
		return m_total_used_memory - std::min(m_total_used_memory, m_pinned_bytes);
	}

	// Diagnostics (the frame graph): an image's cache state, and the other images over its guest
	// range (how many of them hold contents the GPU wrote).
	struct DiagnosticState {
		bool     valid         = false;
		bool     gpu_written   = false; // its contents were written on the GPU
		bool     buffer_write  = false; // GPU buffer writes cover bytes of it
		bool     cpu_dirty     = false; // the guest wrote bytes of it since the last upload
		uint32_t overlaps      = 0;
		uint32_t gpu_overlaps  = 0;
		uint64_t first_overlap = 0; // the guest address of one of them
		// Round 123: the image itself (a binding may view one of another shape).
		GuestRange   data;
		vk::Extent3D extent {};
		uint32_t     levels      = 0;
		uint32_t     layers      = 0;
		uint32_t     host_width  = 0; // its host image's (the backing's)
		uint32_t     host_height = 0;
		vk::Format   format      = vk::Format::eUndefined;
	};
	[[nodiscard]] DiagnosticState DiagnosticImageState(ImageId id);
	// Diagnostics (the frame graph): the pixels the GPU holds for a texture a shader reads, once
	// the commands recorded so far have run, against its guest bytes now (as the CPU wrote them
	// before submitting). A light grid (R32G32_UINT cells: a light-list offset and packed light
	// counts) also gets its lit cells per layer on both sides. Logs "FG 0 grid ...".
	void CompareWithGuest(ImageId id, uint32_t op);
	// A pixel shader's light grid (descriptors.cpp, IsLightGridBinding): its shape (width, height,
	// row pitch) is remembered, and the uploads of R32G32_UINT linear arrays of a shape seen are
	// checked for cells that ask for more than 1024 lights, which game data never does: the
	// shader loops over every one of them, for tens of milliseconds a pixel (GT7's reflection
	// freezes). Such cells are uploaded as empty, unless KYTY_BISECT's grids:raw is on, and logged
	// ("LightGrid:" lines) with the writes the guest memory journal saw there.
	static void NoteLightGridShape(const ImageInfo& info);
	// Diagnostics (the frame graph's verification, FrameGraph::ClaimVerify): an image a shader
	// reads, refreshed as its binding will refresh it, against its guest bytes when the GPU did
	// not write it ("FG <op> stale ..." when they differ), and for an array of six or more layers
	// the contents of each layer as the GPU holds them ("FG <op> layers ...") and, for one the GPU
	// wrote, as guest memory holds them. `what` names the binding in the lines.
	void VerifyWithGuest(ImageId id, uint32_t op, const std::string& what);
	// Diagnostics (KYTY_DUMP_IMAGES, image/imageDump.h): the image as the GPU holds it once the
	// commands recorded so far have run, written to image-dumps/<name>_<address>_<size>.png when
	// the copy lands; once a capture for each content (Image::content_serial). False when it does
	// not apply (multisampled or block-compressed images, formats the dump does not decode).
	bool DumpImage(ImageId id, uint32_t capture, const std::string& name,
	               const std::string& description);
	// A depth image over memory only partly mapped (sparse, in 64 KB blocks), sampled with a
	// depth comparison: the texels of its unmapped blocks are reset to `empty`, the depth that
	// passes the comparison (1.0 for LESS/LEQUAL, 0.0 for GREATER/GEQUAL), once each time its
	// contents change. The console drops the writes that land there and reports reads there as
	// not resident, which the shaders sampling such shadow maps take as lit (TFE, which the
	// translation does not return). Without this the host's texels there, drawn into, shadowed
	// the ground in noisy patches with straight edges. KYTY_SPARSE_HOLES=0 turns it off.
	void ResetUnmappedDepth(ImageId id, float empty);

	// Diagnostics: the KYTY_TRACE_* traces (KYTY_TRACE_DISPATCHES in renderCompute.cpp,
	// KYTY_TRACE_IMAGES here). Lines start "Trace ".
	//
	// Mip 0 of an image as the GPU holds it once the commands recorded so far have run, up to
	// max_bytes: a glyph a layer, logged as "<prefix> ..." when the copy lands. For the float
	// formats light data comes in (B10G11R11, RGBA16F, RG16F, R32F, RG32F, RGBA32F) the glyph
	// tells the light and the layers whose texels are not all finite; for other formats of whole
	// words a texel, only zero or not. False when it does not apply (block-compressed formats,
	// depth, multisampled images).
	bool TraceScanImage(ImageId id, const std::string& prefix, uint64_t max_bytes);
	// VerifyWithGuest outside a frame capture, for an image the GPU did not write: "<label> stale
	// ..." when the GPU's copy differs from its guest bytes. The caller decides how often.
	void TraceVerify(ImageId id, const std::string& label, const std::string& what);
	// GPU thread, at each guest frame end: a snapshot of each image KYTY_TRACE_IMAGES follows
	// that changed in the frame (what changed in it, and who wrote it); and the NaN check
	// (KYTY_NAN_CHECK, default on): the float arrays of six layers or more the GPU writes (light
	// probe atlases, cube maps), up to KYTY_NAN_CHECK_MB (default 256) in all their mips, are
	// scanned on the GPU for texels that are not finite, one image a frame end, each at most every
	// 30 frames and only once it changed; "TextureCache: NaN check ..." is logged at an image's
	// first check and whenever the layers that hold such texels change.
	void TraceFrameEnd();

	// Diagnostics: the words of each layer of each mip of an image whose float bits are infinity
	// or NaN, counted on the GPU (NanScanner) once the commands recorded so far have run; only the
	// counters come back to the host. `done` gets them on the scheduler's priority thread (it must
	// not take the cache's lock). False, and `done` is never called, when the scan does not apply:
	// a format other than the float ones light data comes in (B10G11R11, RG16F, RGBA16F, R32F to
	// RGBA32F), a volume, multisampled or depth image, or one whose mips are over KYTY_GPU_SCAN_MB
	// (default 512) in all.
	struct NonFiniteCounts {
		uint64_t              address      = 0;
		uint32_t              width        = 0;
		uint32_t              height       = 0;
		uint32_t              layers       = 0;
		uint32_t              mips         = 0;
		int                   format       = 0; // the host format (vk::Format)
		uint32_t              guest_format = 0;
		std::vector<uint32_t> words;  // a layer's words, for each mip
		std::vector<uint32_t> counts; // for each mip, a counter a layer

		[[nodiscard]] uint32_t Count(uint32_t mip, uint32_t layer) const {
			return counts[static_cast<size_t>(mip) * layers + layer];
		}
	};
	using NonFiniteDone = std::function<void(NonFiniteCounts&&)>;
	bool ScanNonFiniteGpu(ImageId id, NonFiniteDone done);
	// Changes with each upload and GPU write of the image (Image::content_serial); 0 for an
	// image that is gone.
	[[nodiscard]] uint64_t ContentSerial(ImageId id);
	// The compute shader the GPU thread is recording a dispatch of (names the writers of the images
	// KYTY_TRACE_IMAGES follows; 0 for draws and copies).
	inline static thread_local uint64_t s_trace_shader = 0;
	// KYTY_TRACE_IMAGES: whether it follows any image; and a draw into a render or depth target it
	// follows: the draw's viewport and scissor (`state`), each distinct one once a frame.
	[[nodiscard]] bool TraceEnabled() const noexcept { return m_trace != nullptr; }
	void               TraceTargetDraw(ImageId id, const char* kind, const std::string& state);

private:
	struct TraceState;
	struct TraceHistory;
	// The guest-memory verification behind VerifyWithGuest and TraceVerify: `label` starts the
	// lines ("FG <op>", "Trace dispatch <n>"); `claim` says whether to compare this range now,
	// `claim_layers` the same for an array the GPU wrote.
	void CompareImageWithGuest(ImageId id, const std::string& label, const std::string& what,
	                           bool frame_graph, bool (*claim)(uint64_t, uint64_t),
	                           bool (*claim_layers)(uint64_t, uint64_t));
	// Caller holds m_lock. TraceScanImage's copy and log; with a history, only what changed since
	// the history's snapshot is logged (and the history becomes this one). `header` is logged
	// first.
	bool ScanImageLocked(Image& image, const std::string& prefix, uint64_t max_bytes,
	                     std::shared_ptr<TraceHistory> history, const std::string& header);
	// KYTY_TRACE_IMAGES (caller holds m_lock): whether the image is followed, and its events.
	[[nodiscard]] bool TracesImage(ImageId id) const;
	// Caller holds m_lock. DumpImage's work; `trace` dumps count against the trace's own limit
	// (ImageDump::ClaimTrace) instead of a capture's.
	bool DumpImageLocked(Image& image, bool trace, uint32_t capture, const std::string& name,
	                     const std::string& description);
	// Round 109: DumpImageLocked's work for a block-compressed image (ImageDump::BlockFormat).
	bool DumpBlocksLocked(Image& image, bool trace, uint32_t capture, const std::string& name,
	                      const std::string& description, uint32_t dxgi_format,
	                      uint32_t block_bytes);
	void               TraceImageCreated(ImageId id);
	void               TraceImageEvent(ImageId id, const std::string& event, bool guest_glyphs);
	// Caller holds m_lock: a followed image's binding as a target, once a frame.
	void TraceTargetBound(ImageId id, const ImageDesc& desc, const char* kind);
	// `gpu_write`: a GPU write (CommitGpuWrite), which names the shader whose dispatch it is.
	void TraceImageChanged(Image& image, const char* how, bool gpu_write = false);
	void TraceCpuWrite(ImageId id, const Image& image, uint64_t address, uint64_t size);
	// A followed image right after an upload: its texels that are not finite, on the GPU.
	void TraceUploadScan(ImageId id);
	// Caller holds m_lock: a target's creation inside a known fill (ClearsTargetsFromFills)
	// clears it to the fill; false when nothing was done (the caller uploads as before).
	bool ClearFromKnownFill(ImageId id);
	// Caller holds m_lock: KYTY_ZERO_NEW_MEMORY's clear of a new image that nothing uploads into
	// (Bisect::ZeroNewMemory).
	void ZeroNewImage(ImageId id);

	// The GPU scan's pipeline, scratch and read-back ring (made at the first scan), and the NaN
	// check's state (null with KYTY_NAN_CHECK=0). Caller holds m_lock.
	struct GpuScan;
	struct NanCheck;
	// `max_bytes`: the most it copies of the image (every mip).
	// `scrub` (B10G11R11 and half-float images): the texels that are not finite are also
	// rewritten in the image, a NaN to zero and an infinity to the largest finite value.
	bool ScanNonFiniteGpuLocked(Image& image, NonFiniteDone done, uint64_t max_bytes,
	                            bool scrub = false);
	void NoteNanCheckCandidate(const Image& image);
	void RunNanCheck();

	enum class TransferDirection { Upload, Download };
	struct TextureTransfer;
	struct ImageDownload;

	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint32_t clear_mask = UINT32_MAX;
		// The last fill's value (ClearMeta), when known.
		uint32_t fill_value = 0;
		bool     fill_known = false;
		// The HTile's size (its depth target's last binding), 0 when not known.
		uint64_t size = 0;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
		// The cached image stays beside an image of the requested kind (KeepReadOnlyAlias).
		bool kept_alias = false;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// The registered images of each page of the image page table: written under m_lock, read
	// without it, so a CPU write fault on memory without images skips the lock (NeySlim's
	// "skip texture invalidation on pages without images", KytyPS5 PR 1284).
	class RegisteredPageCounts final {
	public:
		RegisteredPageCounts() = default;
		~RegisteredPageCounts();
		KYTY_CLASS_NO_COPY(RegisteredPageCounts);

		void               Add(size_t page, int32_t delta);
		[[nodiscard]] bool Any(size_t first, size_t last_exclusive) const noexcept;

	private:
		static constexpr size_t LeafBits = 10;
		static constexpr size_t LeafSize = size_t {1} << LeafBits;
		using Leaf                       = std::array<std::atomic<uint32_t>, LeafSize>;

		static constexpr size_t LeafCount = ImagePageTable::kPageCount / LeafSize;
		// On the heap: 128 KB, too much for a TextureCache on a thread's stack (the tests').
		std::unique_ptr<std::atomic<Leaf*>[]> m_leaves =
		    std::make_unique<std::atomic<Leaf*>[]>(LeafCount);
	};
	// Whether an image may lie in [address, address + size): false is exact.
	[[nodiscard]] bool MayHaveImages(uint64_t address, uint64_t size) const noexcept;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
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

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      ReleaseImage(ImageId id);
	// `why` is logged when the image held pixels the GPU wrote that guest memory does not and
	// that were not copied elsewhere (`copied`): they are lost. And for an image KYTY_TRACE_IMAGES
	// follows.
	void                      FreeImage(ImageId id, const char* why = nullptr, bool copied = false);
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

	// Caller holds m_lock: MayReuseTextureBinding's checks of an image.
	[[nodiscard]] bool ReusableTextureImage(const Image& image) const;
	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	// Round 79: whether a texture read of `requested` leaves `cached` (at the same address, with
	// the same block extent and size) in place beside an image of its own, instead of replacing
	// it: both small, single-sampled and never written by the GPU, and of kinds no one native
	// image serves (a volume and a flat image, a depth read of a color image). GT7 reads a 1x1
	// placeholder as R8, R16 and D16, flat and as a volume: each read of another kind replaced
	// the image before, 250 to 500 times a frame. Caller holds m_lock.
	[[nodiscard]] bool        KeepReadOnlyAlias(const ImageInfo& requested, BindingType binding,
	                                            const Image& cached) const;
	[[nodiscard]] static bool ReadOnlyAliasCandidate(const Image& image);
	void                      MarkReadOnlyAlias(Image& image);
	// A GPU write to `image`: the read-only aliases that start where it does are freed (their
	// bytes are no longer what the memory holds). Caller holds m_lock.
	void                        FreeReadOnlyAliasesOf(const Image& image);
	void                        RefreshImage(ImageId id);
	// Small linear images the CPU writes, compared with their guest bytes instead of trusting
	// the page tracking alone (KYTY_GUEST_HASH_IMAGES_KB).
	[[nodiscard]] bool ChecksGuestContent(const Image& image) const;
	void               RefreshByContent(ImageId id);
	// The GPU-side MaterializeColorClear of GPU-written render-target metadata; false when it
	// does not apply (the caller then reads the metadata back).
	void                        DescribeUintArrayImage(ImageId id, const ImageDesc& desc);
	// `unwritten`: nothing wrote the metadata since its last check (a recheck, for the outcome's
	// report); `check` gets the first slice's check number.
	bool MaterializeColorClearOnGpu(ImageId id, const ImageDesc& desc, uint32_t first,
	                                uint32_t image_first, uint32_t count, uint64_t slice_size,
	                                bool unwritten, uint32_t* check);
	[[nodiscard]] static bool   RechecksColorClears();
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	// InitializeImage: a light grid's upload source before the copy is recorded
	// (NoteLightGridShape).
	void CheckLightGridUpload(const Image& image, Buffer& source, uint64_t source_offset);
	[[nodiscard]] TextureTransfer
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] ImageDownload BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, ImageDownload transfer);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	// evicting: the collector writes the image back before freeing it; the write is dropped if
	// the guest bytes change before it lands (KYTY_EVICT_WRITEBACK_GUARD).
	// `landed`, when given, is set once the write-back has reached guest memory (or was dropped
	// because the guest wrote the bytes first): an evicted image is freed only then.
	[[nodiscard]] bool DownloadImageMemory(ImageId id, bool evicting = false,
	                                       std::shared_ptr<std::atomic<bool>>* landed = nullptr);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock m_lock {Common::PerfStats::Lock::TextureCache};
	// Registrations and unregistrations of images (ImageSetGeneration), and where they were
	// (RangeStamp).
	std::atomic<uint64_t> m_image_set_generation {1};
	ImageZoneStamps                                   m_zone_stamps;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	// Render-target fast clears decided on the GPU (MaterializeColorClear); made at first use.
	std::unique_ptr<ConditionalColorClear>            m_conditional_clear;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	RegisteredPageCounts                                          m_registered_pages;
	std::map<std::pair<vk::Format, Prospero::ImageType>, ImageId> m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	// SingleSampleDepth's copies, by the multisample depth image they copy: the copy (an image
	// with no guest memory, so never registered) and the depth image's content serial it holds.
	struct SingleSampleDepthCopy {
		ImageId  copy {};
		uint64_t serial = UINT64_MAX;
	};
	std::unordered_map<ImageId, SingleSampleDepthCopy> m_single_sample_depths;
	// SingleSampleStencil's, likewise (the serial: the depth image's and its stencil planes').
	std::unordered_map<ImageId, SingleSampleDepthCopy> m_single_sample_stencils;
	// SubExtentCopy's, by the image they copy (a few extents each): likewise never registered.
	struct SubExtentCopyEntry {
		ImageId  copy {};
		uint32_t width  = 0;
		uint32_t height = 0;
		uint64_t serial = UINT64_MAX;
		uint64_t used   = 0;
	};
	std::unordered_map<ImageId, std::vector<SubExtentCopyEntry>> m_sub_extent_copies;
	std::unordered_set<ImageId>                       m_bindless_evictable;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	// The addresses color targets' CMASK and DCC start at (IsColorMeta).
	std::unordered_set<uint64_t> m_color_metas;
	// Image write-backs still to land, one entry each (PendingWriteBacks); shared with their
	// publications on the priority thread, which remove their own.
	struct PendingWriteBackList {
		mutable std::mutex                         mutex;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	std::shared_ptr<PendingWriteBackList> m_pending_writebacks =
	    std::make_shared<PendingWriteBackList>();
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	// The thresholds as configured; the ones above come down from them while memory is short
	// (GraphicContext::MemoryShortfall: device use past 90 % of the video memory target).
	uint64_t                                          m_base_trigger_gc_memory  = 0;
	uint64_t                                          m_base_pressure_gc_memory = 0;
	uint64_t                                          m_base_critical_gc_memory = 0;
	// What the collector last wrote into the thresholds: values that differ were set anew
	// (the constructor's caps, a test's configuration) and become the base.
	uint64_t                                          m_applied_trigger_gc_memory  = UINT64_MAX;
	uint64_t                                          m_applied_pressure_gc_memory = UINT64_MAX;
	uint64_t                                          m_applied_critical_gc_memory = UINT64_MAX;
	// Engine method: images the guest dropped from its texture heaps (no key refers to them any
	// more), freed once no draw has used them for a few frames, whatever the memory pressure.
	uint64_t                                          m_pinned_bytes = 0;
	uint64_t                                          m_pinned_frame = UINT64_MAX;
	std::vector<ImageId>                              m_guest_dropped;
	uint64_t                                          m_guest_dropped_freed       = 0;
	uint64_t                                          m_guest_dropped_freed_bytes = 0;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	// Evictions are budgeted per frame, not per collector run: see RunGarbageCollector.
	uint64_t         m_gc_budget_frame        = 0;
	uint64_t         m_gc_freed_bytes_frame   = 0;
	size_t           m_gc_freed_images_frame  = 0;
	uint64_t         m_gc_written_back_bytes_frame = 0;
	// Two-phase eviction of GPU-written images: the write-back is deferred until the GPU is done,
	// and the image stays registered until it has landed. Freeing it at once let a draw in between
	// recreate it from the old guest bytes, without what the GPU had drawn into it (the glyph
	// atlases: "half of the text is missing", 2026-10-09). One used again in the meantime stays.
	struct PendingEviction {
		ImageId                            id;
		uint64_t                           address = 0; // the slot may be reused: checked
		uint64_t                           frame   = 0; // frame_accessed_last at eviction
		std::shared_ptr<std::atomic<bool>> landed;
	};
	std::vector<PendingEviction> m_pending_evictions;
	std::unordered_set<ImageId>  m_evicting;
	// The LRU clock: presented frames, advanced by the collector's own ticks as well so a
	// stretch without presents (a loading screen) still ages its entries.
	[[nodiscard]] uint64_t LruClock() const noexcept;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_readback_linear_images = false;
	// Registered images kept as read-only aliases (KeepReadOnlyAlias); m_lock.
	uint64_t m_read_only_aliases = 0;
	// KYTY_TRACE_IMAGES (null when it is not set).
	std::unique_ptr<TraceState> m_trace;
	std::unique_ptr<GpuScan>    m_gpu_scan;
	std::unique_ptr<NanCheck>   m_nan_check;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
