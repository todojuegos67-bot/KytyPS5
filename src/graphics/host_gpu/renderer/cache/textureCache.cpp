#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <unordered_set>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;
// A fill counts for a depth target registered within this many submissions: about two frames.
constexpr uint64_t UnclaimedFillTicks = 64;
constexpr size_t   UnclaimedFillsMax  = 64;

static bool InPlaceResizeEnabled() {
	static const bool enabled = std::getenv("KYTY_DEBUG_NO_INPLACE_RESIZE") == nullptr;
	return enabled;
}

// KYTY_GPU_ZONES: describe an image the first time it is refreshed for each cause, so the guest
// addresses that key its zone rows can be matched to a size, format, and writer.
// KYTY_GPU_ZONES: the binding whose lookup is refreshing an image, for the refresh log.
thread_local const char* t_refresh_via = "other";

class RefreshVia final {
public:
	explicit RefreshVia(const char* via) noexcept: m_previous(t_refresh_via) { t_refresh_via = via; }
	~RefreshVia() { t_refresh_via = m_previous; }
	RefreshVia(const RefreshVia&)            = delete;
	RefreshVia& operator=(const RefreshVia&) = delete;

private:
	const char* m_previous;
};

void LogZoneRefresh(const Image& image) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	const bool     buffer_modified = image.IsBufferModified();
	const auto&    info            = image.info;
	const uint64_t key             = info.data.address ^ (buffer_modified ? 1u : 0u) ^
	                     (static_cast<uint64_t>(t_refresh_via[0]) << 56u);
	{
		std::scoped_lock lock {mutex};
		if (!seen.insert(key).second) {
			return;
		}
	}
	std::printf("gpu-zones: refresh image 0x%016" PRIx64 " size=0x%" PRIx64
	            " %ux%ux%u guest_format=%u tile=%u levels=%u layers=%u cause=%s via=%s\n",
	            info.data.address, info.data.size, info.extent.width, info.extent.height,
	            info.extent.depth, static_cast<uint32_t>(info.guest_format),
	            static_cast<uint32_t>(info.tile_mode), info.resources.levels, info.resources.layers,
	            buffer_modified ? "gpu-buffer-write" : "cpu-write", t_refresh_via);
}

// KYTY_GPU_ZONES: the first time each kind of GPU buffer write lands on an image, print it with
// how much of the image it covers and the PM4 packet that caused it.
void LogZoneBufferWrite(const Image& image, uint64_t address, uint64_t size,
                        TextureCache::GpuWriteSource source) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	const auto&                         info = image.info;
	const uint64_t key = info.data.address ^ ((static_cast<uint64_t>(source) + 1u) << 60u);
	{
		std::scoped_lock lock {mutex};
		if (!seen.insert(key).second) {
			return;
		}
	}
	const char* kind = source == TextureCache::GpuWriteSource::Fill   ? "fill"
	                   : source == TextureCache::GpuWriteSource::Copy ? "copy"
	                                                                  : "shader-store";
	const bool  whole = address <= info.data.address &&
	                   address + size >= info.data.address + info.data.size;
	std::printf("gpu-zones: buffer %s on image 0x%016" PRIx64 " %ux%u guest_format=%u tile=%u "
	            "write=0x%016" PRIx64 "+0x%" PRIx64 " %s pm4=0x%x image_gpu_modified=%d\n",
	            kind, info.data.address, info.extent.width, info.extent.height,
	            static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
	            address, size, whole ? "whole-image" : "partial", DrainStats::t_pm4_op,
	            image.IsGpuModified() ? 1 : 0);
}

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.clear_register_valid &&
		       DecodePackedColorClear(format, metadata.clear_word, clear);
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache), m_dcc_resolver(graphics, scheduler),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()) {
	m_dcc_gpu_clear = Config::DccGpuClearEnabled() && m_dcc_resolver.Available();
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		// On 8GB GPUs (e.g. RX 6600 XT), keep working sets resident (up to ~5.5-6GB) without thrashing
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(budget - GiB / 2, 3 * GiB));
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(budget - GiB - GiB / 4, 2 * GiB));
		m_trigger_gc_memory  = static_cast<uint64_t>(
		    std::max<int64_t>(budget - 2 * GiB, GiB));
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameImageInfo(const ImageInfo& a, const ImageInfo& b) noexcept {
	return a.data == b.data && a.stencil == b.stencil && a.metadata == b.metadata &&
	       a.htile_clear_mask == b.htile_clear_mask && a.pixel_format == b.pixel_format &&
	       a.guest_format == b.guest_format && a.type == b.type && a.extent == b.extent &&
	       a.resources == b.resources && a.pitch == b.pitch &&
	       a.bytes_per_block == b.bytes_per_block && a.samples == b.samples &&
	       a.tile_mode == b.tile_mode && a.bgra16 == b.bgra16 && a.mip_layout == b.mip_layout;
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

// Debugging aid: KYTY_DEBUG_IMAGE_CHURN=1 prints every image the cache creates or frees, with the
// path that did it (ChurnReason), to find images recreated in steady state.
namespace {
thread_local const char* t_churn_reason = "other";
struct ChurnReason {
	explicit ChurnReason(const char* reason): previous(t_churn_reason) { t_churn_reason = reason; }
	~ChurnReason() { t_churn_reason = previous; }
	ChurnReason(const ChurnReason&)            = delete;
	ChurnReason& operator=(const ChurnReason&) = delete;
	const char*  previous;
};
void LogImageChurn(const char* what, const Image& image) {
	static const bool enabled = std::getenv("KYTY_DEBUG_IMAGE_CHURN") != nullptr;
	if (!enabled) {
		return;
	}
	const auto& info = image.info;
	std::printf("image-churn: %s %s 0x%016" PRIx64 " size=0x%" PRIx64
	            " %ux%ux%u fmt=%u guest=%u type=%u tile=%u mips=%u layers=%u samples=%u"
	            " usage=%s%s%s%s\n",
	            what, t_churn_reason, info.data.address, info.data.size, info.extent.width,
	            info.extent.height, info.extent.depth, static_cast<uint32_t>(info.pixel_format),
	            static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.type),
	            static_cast<uint32_t>(info.tile_mode), info.resources.levels, info.resources.layers,
	            info.samples, image.usage.texture ? "T" : "", image.usage.render_target ? "R" : "",
	            image.usage.depth_target ? "D" : "", image.usage.storage ? "S" : "");
}
} // namespace

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	LogImageChurn("new", m_slot_images[id]);
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	return id;
}

// Callers hold m_lock, so there is one writer at a time. The regions are stamped first and the
// new generation published after them: a reader that sees a generation also sees every stamp up
// to it (RangeGeneration's memo relies on this).
void TextureCache::AdvanceImageSetGeneration(GuestRange range) noexcept {
	const auto stamp = m_image_set_generation.load(std::memory_order_relaxed) + 1;
	const auto first = range.address >> RegionGenerationBits;
	const auto last  = (range.End() - 1) >> RegionGenerationBits;
	for (auto region = first; region <= last; region++) {
		m_region_generations[region % RegionGenerationBuckets].store(stamp,
		                                                            std::memory_order_relaxed);
	}
	m_image_set_generation.store(stamp, std::memory_order_release);
}

// KYTY_DEBUG_AB=regiongen uses the whole image set's generation for every range in every other
// window. The two agree on what they call unchanged: the whole set's generation is at least every
// range's, and equals a range's only when that range saw the newest change.
uint64_t TextureCache::RangeGeneration(GuestRange range) const noexcept {
	static const bool ab    = AbSelected("regiongen");
	const auto        whole = m_image_set_generation.load(std::memory_order_acquire);
	if ((ab && AbFeatureOff()) || range.size == 0) {
		return whole;
	}
	// Draws ask about the same ranges draw after draw, and while the whole set's generation is
	// unchanged no range's is: the answer is kept with the generation it was computed under. A
	// render target spans many regions, and their buckets are scattered over the table.
	// KYTY_DEBUG_AB=rangememo computes every answer in every other window.
	struct Memo {
		const TextureCache* cache      = nullptr;
		uint64_t            address    = 0;
		uint64_t            size       = 0;
		uint64_t            whole      = 0;
		uint64_t            generation = 0;
	};
	thread_local std::array<Memo, 1024> memos {};
	static const bool memo_ab = AbSelected("rangememo");
	auto& memo = memos[((range.address >> 12u) ^ (range.address >> 24u) ^ range.size) % memos.size()];
	if (memo.cache == this && memo.whole == whole && memo.address == range.address &&
	    memo.size == range.size && !(memo_ab && AbFeatureOff())) {
		return memo.generation;
	}
	uint64_t   generation = m_image_set_base;
	const auto first      = range.address >> RegionGenerationBits;
	const auto last       = (range.End() - 1) >> RegionGenerationBits;
	for (auto region = first; region <= last; region++) {
		generation = std::max(generation, m_region_generations[region % RegionGenerationBuckets].load(
		                                      std::memory_order_relaxed));
	}
	memo = {this, range.address, range.size, whole, generation};
	return generation;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_image_page_table[page].push_back(id);
	});
	AdvanceImageSetGeneration(image.info.data);
	image.registered = true;
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	image.lru_tick   = m_gc_tick;
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
	});
	AdvanceImageSetGeneration(image.info.data);
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id, std::vector<ImageId>* retired) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	if (!image->depth_id) {
		std::vector<ImageId> associations;
		m_slot_images.ForEach([&](ImageId candidate, const Image& associated) {
			if (associated.depth_id == id) {
				associations.push_back(candidate);
			}
		});
		for (const auto association: associations) {
			FreeImage(association, retired);
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image->info.HasMetadata()) {
		const auto meta_address = image->info.metadata.range.address;
		const auto metadata     = m_surface_metas.find(meta_address);
		// A depth target resized in place at the same allocation keeps the HTile address: the
		// entry, and its clear state, belong to the replacement too.
		const auto shares_htile = [&] {
			for (const auto other_id:
			     FindImagesInRegion(image->info.data.address, image->info.data.size, false)) {
				const auto* other = m_slot_images.try_get(other_id);
				if (other_id != id && other != nullptr && other->registered &&
				    other->info.metadata.kind == ImageMetadataKind::Htile &&
				    other->info.metadata.range.address == meta_address) {
					return true;
				}
			}
			return false;
		};
		if (metadata != m_surface_metas.end() &&
		    image->info.metadata.kind == ImageMetadataKind::Htile &&
		    metadata->second.type == MetaDataInfo::Type::HTile && !shares_htile()) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
			m_surface_meta_generation.fetch_add(1, std::memory_order_release);
		}
	}
	UnregisterImage(id);
	if (retired != nullptr) {
		retired->push_back(id);
	} else if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_images.erase(id); });
	} else {
		m_slot_images.erase(id);
	}
}

void TextureCache::FreeImage(ImageId id, std::vector<ImageId>* retired) {
	auto* image = m_slot_images.try_get(id);
	if (image != nullptr) {
		LogImageChurn("free", *image);
	}
	while (image != nullptr && image->registered &&
	       HasPendingDownload(image->info.data.address, image->info.data.size)) {
		// Overlap replacement also retires images outside the collector. Preserve its
		// registration and write watcher until the last overlapping publication completes.
		// Wait does not run general callbacks, so native resources remain alive throughout.
		const auto completion_tick = m_scheduler.CurrentTick();
		{
			struct ReacquireLock {
				explicit ReacquireLock(TrackingSpinLock& mutex): mutex(mutex) { mutex.unlock(); }
				~ReacquireLock() { mutex.lock(); }
				TrackingSpinLock& mutex;
			};
			// All callers own m_lock. Release it while waiting, including for a priority
			// callback that needs the cache; restore it before touching the owner again.
			const ReacquireLock     unlocked(m_lock);
			DrainStats::ReasonScope reason(DrainStats::Reason::FreeImage);
			m_scheduler.Wait(completion_tick);
			m_scheduler.WaitPriorityOperations(completion_tick);
		}
		image = m_slot_images.try_get(id);
	}
	if (image == nullptr || !image->registered) {
		return;
	}
	if (image->IsGpuModified()) {
		image->ClearGpuModified();
	}
	DeleteImage(id, retired);
}

void TextureCache::TouchImage(Image& image) {
	// Draws touch the same images many times per GC tick: the LRU holds this tick already.
	if (image.registered && image.lru_tick != m_gc_tick) {
		image.lru_tick = m_gc_tick;
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	if (image.NeedsMaybeCpuHash()) {
		image.SetMaybeCpuHash(image.HashGuestEdges());
	}
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	if (m_readback_linear_images && !image.info.IsTiled() && !image.info.data.Empty()) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
		}
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr) {
				return;
			}
			if (image->query_epoch == query_epoch) {
				return;
			}
			image->query_epoch = query_epoch;
			if (image->Overlaps(address, size, page_overlap)) {
				result.push_back(id);
			}
		});
	});
	return result;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	const auto format = desc.info.pixel_format;
	if (const auto found = m_null_images.find(format); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(format, id);
	return id;
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	const bool source_depth      = source.info.IsDepth();
	const bool destination_depth = destination.info.IsDepth();
	if (source_depth == destination_depth) {
		return false;
	}
	auto&      depth          = source_depth ? source : destination;
	auto&      color          = source_depth ? destination : source;
	const auto transfer_bytes = DepthAspectTransferBytes(depth.backing.format);
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t) || transfer_bytes != sizeof(uint32_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * transfer_bytes;
	const uint64_t color_slice =
	    static_cast<uint64_t>(color.info.pitch) * color.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || color_slice > UINT64_MAX / layers);
	const auto                       depth_size = depth_slice * layers;
	const auto                       color_size = color_slice * layers;
	const std::array                 depth_copies {
	    vk::BufferImageCopy {0,
	                         depth.info.pitch,
	                         depth.info.extent.height,
	                         {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                         {},
	                         depth.info.extent}};
	const std::array color_copies {
	    vk::BufferImageCopy {0,
	                         color.info.pitch,
	                         color.info.extent.height,
	                         {vk::ImageAspectFlagBits::eColor, 0, 0, layers},
	                         {},
	                         color.info.extent}};

	auto                         depth_buffer = m_tiler.GetScratchBuffer(depth_size);
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(color.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * transfer_bytes,
	    .source_slice_stride = color_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_depth) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, color_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(color_copies, color_buffer.buffer, color_buffer.offset,
		                   color_buffer.size);
	} else {
		source.Download(color_copies, color_buffer.buffer, color_buffer.offset, color_buffer.size);
		m_tiler.ConvertD16(color_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	GpuZones::KeyScope zone_key(destination.info.data.address);
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		MarkGpuModified(destination);
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	GpuZones::KeyScope zone_key(destination.info.data.address);
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		MarkGpuModified(destination);
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	ChurnReason reason("depth-overlap");
	auto&       cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	const bool raw_d16_texture =
	    binding == BindingType::Texture && cached.info.IsDepth() &&
	    cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt &&
	    requested.pixel_format == vk::Format::eR16Uint && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data == cached.info.data &&
	    requested.extent == cached.info.extent && requested.resources == cached.info.resources &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout          = cached.info.mip_layout;
		info.tiling              = cached.info.tiling;
		info.tiled_slice_stride  = cached.info.tiled_slice_stride;
		info.linear_slice_stride = cached.info.linear_slice_stride;
		info.first_tail_level    = cached.info.first_tail_level;
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr) {
		return {merged_id};
	}
	auto&      cached       = *owner;
	const auto current_tick = m_scheduler.CurrentTick();
	const bool safe_to_delete =
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval;

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     !std::equal(requested.mip_layout.begin(), requested.mip_layout.end(),
		                 cached.info.mip_layout.begin(),
		                 [](const ImageMipInfo& left, const ImageMipInfo& right) {
			                 return left.offset == right.offset && left.size == right.size &&
			                        left.pitch == right.pitch && left.height == right.height &&
			                        left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
			                        left.surface_z == right.surface_z;
		                 }))) {
			if (safe_to_delete) {
				ChurnReason reason("overlap-layout");
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			const auto result    = m_slot_images.try_get(result_id);
			return {result != nullptr && ImageViewOps::FormatsCompatible(result->info.pixel_format,
			                                                             requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	// A surface regrown in place (dynamic resolution) at an unchanged row pitch keeps every texel
	// at its guest address, so the game can read back what it wrote at the smaller extent.
	if (!merged_id && InPlaceResizeEnabled() && requested.data == cached.info.data &&
	    requested.extent != cached.info.extent &&
	    requested.extent.width >= cached.info.extent.width &&
	    requested.extent.height >= cached.info.extent.height &&
	    requested.extent.depth == cached.info.extent.depth && requested_block == cached_block &&
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.type == cached.info.type && !requested.IsVolume() &&
	    requested.resources == cached.info.resources && requested.resources.levels == 1 &&
	    requested.resources.layers == 1 && requested.tile_mode == cached.info.tile_mode &&
	    !requested.IsDepth() && !cached.info.IsDepth() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata() &&
	    requested.GetColorTransform() == cached.info.GetColorTransform() &&
	    requested.mip_layout[0].offset == cached.info.mip_layout[0].offset &&
	    requested.mip_layout[0].pitch == cached.info.mip_layout[0].pitch &&
	    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
	    cached.IsGpuModified()) {
		return {ExpandImage(requested, cached_id)};
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		ChurnReason reason("overlap-merge");
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		ChurnReason reason("overlap-replace");
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	ChurnReason reason("expand");
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	return expanded_id;
}

bool TextureCache::CanDownload(const Image& image) const {
	const auto& info = image.info;
	if (info.samples != 1 || image.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0) {
		return false;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		return IsSupportedDepthPlaneReadback(info) &&
		       Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
	}
	return info.metadata.compression == VideoOutCompression::Uncompressed &&
	       (!info.IsTiled() || !Prospero::IsFmaskTextureFormat(info.guest_format)) &&
	       info.LinearSize() != 0;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	GpuZones::KeyScope zone_key(image.info.data.address);
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto& info        = image.info;
	const auto  binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	if (binding != BindingType::DepthTarget) {
		if ((binding == BindingType::RenderTarget || binding == BindingType::VideoOut) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		if (binding == BindingType::VideoOut &&
		    info.metadata.compression != VideoOutCompression::Uncompressed) {
			EXIT("TextureCache: invalid video-out upload\n");
		}
		const auto linear = m_tiler.Detile(source.Handle(), source_offset, info);
		const auto copies = info.BufferCopies(linear.offset);
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
		return;
	}

	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto layers          = info.resources.layers;
	const auto full_slice_size = info.linear_slice_stride;
	auto       linear          = m_tiler.Detile(source.Handle(), source_offset, info);
	auto       copies          = info.BufferCopies(linear.offset);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		copies.resize(1);
		copies[0].bufferOffset      = linear.offset;
		copies[0].bufferRowLength   = info.pitch;
		copies[0].bufferImageHeight = info.extent.height;
		copies[0].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, 0, layers};
		copies[0].imageExtent       = info.extent;
	}
	destination.Upload(copies, linear.buffer, linear.offset, linear.size);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.info.data.Empty()) {
		return;
	}
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		RecordImageUpload(image.info.data.address, image.info.data.size, image.info.extent.width,
		                  image.info.extent.height, static_cast<uint32_t>(image.info.guest_format),
		                  static_cast<uint32_t>(image.info.tile_mode), image.IsBufferModified());
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

TextureCache::TargetState TextureCache::CurrentTargetState() const noexcept {
	return {m_image_set_generation.load(std::memory_order_acquire), m_scheduler.CurrentTick(),
	        m_gc_tick};
}

// Whether MaterializeColorClearNow would change nothing, as it did last time for this image: the
// same surface and view, no clear flag registered since (its erase would find none), and for a
// GPU-written check, no GPU-dirty change and no GPU write into checked slices since (the same
// slices stay checked, with the same keys); for a check of metadata the GPU had not written, no
// GPU-dirty change since and the same first key in each slice.
bool TextureCache::ColorClearUnchanged(const Image& image, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) const {
	const auto& check = image.color_clear_check;
	return check.valid && check.metadata == desc.info.metadata &&
	       image.info.metadata == desc.info.metadata && check.view == desc.view_info &&
	       check.resources == desc.info.resources && check.extent == desc.info.extent &&
	       check.image_type == static_cast<uint32_t>(desc.info.type) &&
	       check.metadata_base_layer == metadata_base_layer &&
	       check.binding_type == static_cast<uint8_t>(desc.type) &&
	       check.surface_meta_generation ==
	           m_surface_meta_generation.load(std::memory_order_relaxed) &&
	       (!check.gpu_checked ||
	        (check.gpu_dirty_generation == m_buffer_cache.GpuDirtyGeneration() &&
	         check.dcc_checked_generation ==
	             m_dcc_checked_generation.load(std::memory_order_acquire))) &&
	       (!check.cpu_checked ||
	        (check.gpu_dirty_generation == m_buffer_cache.GpuDirtyGeneration() &&
	         ColorClearKeysUnchanged(check.keys)));
}

bool TextureCache::IsColorClearCurrent(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) const {
	return (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	        desc.info.metadata.kind != ImageMetadataKind::Cmask) ||
	       ColorClearUnchanged(m_slot_images[id], desc, metadata_base_layer);
}

// Whether each slice's first metadata key still reads as it did (see ColorClearCheck::cpu_checked).
bool TextureCache::ColorClearKeysUnchanged(const Image::ColorClearCheck::Keys& keys) {
	for (uint32_t i = 0; i < keys.count; i++) {
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(keys.addresses[i], &code, sizeof(code)) ||
		    code != keys.codes[i]) {
			return false;
		}
	}
	return true;
}

// Most lookups bind a color surface whose metadata was already checked, with nothing to apply:
// skip the check while nothing it depends on has changed (ColorClearUnchanged). Only Thread_Gpu
// changes images and their checks. KYTY_VERIFY_COLOR_CLEAR=1 checks anyway and reports skips that
// would have missed a change; KYTY_DEBUG_AB=colorclear checks every time in alternate windows.
void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	static const bool verify = std::getenv("KYTY_VERIFY_COLOR_CLEAR") != nullptr;
	static const bool ab     = AbSelected("colorclear");
	const bool        skip   = !(ab && AbFeatureOff()) &&
	                  ColorClearUnchanged(m_slot_images[id], desc, metadata_base_layer);
	if (skip && !verify) {
		return;
	}
	bool                         changed = false;
	Image::ColorClearCheck::Keys keys;
	const auto outcome = MaterializeColorClearNow(id, desc, metadata_base_layer, changed, keys);
	if (skip) {
		static std::atomic<uint64_t> checked {0};
		static std::atomic<uint64_t> missed {0};
		const auto count = checked.fetch_add(1, std::memory_order_relaxed) + 1;
		if (changed || outcome == ColorClearOutcome::Other) {
			const auto index = missed.fetch_add(1, std::memory_order_relaxed);
			if (index < 16) {
				std::printf("color-clear verify: skip of 0x%016" PRIx64 " (metadata 0x%016" PRIx64
				            ") would have missed a change (outcome %d, changed %d)\n",
				            m_slot_images[id].info.data.address, desc.info.metadata.range.address,
				            static_cast<int>(outcome), changed ? 1 : 0);
			}
		}
		if (count % 100000 == 0) {
			std::printf("color-clear verify: skips=%" PRIu64 " missed=%" PRIu64 "\n", count,
			            missed.load(std::memory_order_relaxed));
			std::fflush(stdout);
		}
	}
	auto& check = m_slot_images[id].color_clear_check;
	if (outcome == ColorClearOutcome::Other) {
		check.valid = false;
		return;
	}
	check.metadata                = desc.info.metadata;
	check.view                    = desc.view_info;
	check.resources               = desc.info.resources;
	check.extent                  = desc.info.extent;
	check.image_type              = static_cast<uint32_t>(desc.info.type);
	check.metadata_base_layer     = metadata_base_layer;
	check.binding_type            = static_cast<uint8_t>(desc.type);
	check.valid                   = true;
	check.gpu_checked             = outcome == ColorClearOutcome::GpuChecked;
	check.cpu_checked             = outcome == ColorClearOutcome::CpuChecked;
	check.keys                    = keys;
	check.surface_meta_generation = m_surface_meta_generation.load(std::memory_order_relaxed);
	check.gpu_dirty_generation    = m_buffer_cache.GpuDirtyGeneration();
	check.dcc_checked_generation  = m_dcc_checked_generation.load(std::memory_order_acquire);
}

TextureCache::ColorClearOutcome
TextureCache::MaterializeColorClearNow(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer, bool& changed,
                                       Image::ColorClearCheck::Keys& keys) {
	keys.count = 0;
	const auto range = desc.info.metadata.range;
	{
		std::scoped_lock lock {m_lock};
		auto& image = m_slot_images[id];
		changed |= !(image.info.metadata == desc.info.metadata);
		// The overlap rules read whether an image has metadata (see OverlapLookup).
		if (image.registered && image.info.HasMetadata() != desc.info.HasMetadata()) {
			AdvanceImageSetGeneration(image.info.data);
		}
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile/CMask/FMask clear flag.
		if (m_surface_metas.erase(range.address) != 0) {
			changed = true;
			m_surface_meta_generation.fetch_add(1, std::memory_order_release);
		}
		if (DrainStats::Enabled() && range.Valid()) {
			m_dcc_metadata_seen.Add(range.address, range.size);
		}
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return ColorClearOutcome::NoSlices;
		}
	}
	const auto layers = desc.info.TransferLayers();
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	const auto slice_size  = range.size / layers;
	const bool gpu_written = m_buffer_cache.IsRegionGpuModified(range.address, range.size);
	if (gpu_written) {
		bool no_op = false;
		if (MaterializeDccClearOnGpu(id, desc, range.address + slice_size * first, slice_size,
		                             image_first, count, &no_op)) {
			return no_op ? ColorClearOutcome::GpuChecked : ColorClearOutcome::Other;
		}
	}
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	if (gpu_written) {
		DrainStats::ReasonScope reason(DrainStats::Reason::DccClear);
		m_buffer_cache.ReadMemory(range.address, range.size, false);
	}
	uint64_t cleared_slices = 0;
	struct CheckRecord {
		bool      record;
		uint64_t& cleared;
		~CheckRecord() {
			if (record) {
				DrainStats::ReasonScope reason(DrainStats::Reason::DccClear);
				DrainStats::Record(DrainStats::Kind::DccCheck, cleared);
			}
		}
	} check_record {gpu_written, cleared_slices};
	// Metadata the GPU has not written, whose first keys are no clear, gets the same result while
	// those keys stay (see ColorClearCheck::cpu_checked).
	bool settled = !gpu_written && count <= keys.addresses.size();
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		if (settled) {
			keys.addresses[keys.count] = address;
			keys.codes[keys.count]     = code;
			keys.count++;
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		// The slice's other keys decide, and they can change without this one.
		settled = false;
		std::vector<uint8_t> bytes(slice_size);
		if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
			EXIT("TextureCache: failed to read color metadata slice\n");
		}
		if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
			continue;
		}
		{
			std::scoped_lock lock {m_lock};
			ClearImage(m_scheduler.Current(), id, view.format,
			           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		cleared_slices++;
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, slice_size);
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
	}
	if (settled && cleared_slices == 0) {
		return ColorClearOutcome::CpuChecked;
	}
	keys.count = 0;
	return ColorClearOutcome::Other;
}

bool TextureCache::MaterializeDccClearOnGpu(ImageId id, const ImageDesc& desc,
                                            uint64_t slices_address, uint64_t slice_size,
                                            uint32_t image_first, uint32_t count, bool* no_op) {
	if (!m_dcc_gpu_clear || count == 0) {
		return false;
	}
	GpuZones::KeyScope zone_key(m_slot_images[id].info.data.address);
	std::array<vk::ClearColorValue, DccClearResolver::CodeCount> colors {};
	uint32_t                                                     code_mask = 0;
	for (uint32_t index = 0; index < DccClearResolver::CodeCount; index++) {
		if (DecodeColorClear(desc, DccClearResolver::Code(index), colors[index])) {
			code_mask |= 1u << index;
		}
	}
	if (code_mask == 0) {
		// No key clears this view, so the metadata bytes cannot change the image.
		if (no_op != nullptr) {
			*no_op = true;
		}
		return true;
	}
	// A GPU check consumed every clear it found, and no GPU write has touched the slices since,
	// so any binding type (textures, storage, video out) would find no clear to apply either.
	if (DccSlicesChecked(slices_address, slice_size, count, code_mask)) {
		if (no_op != nullptr) {
			*no_op = true;
		}
		return true;
	}
	// Render targets and sampled textures: both consume the keys (as the CPU path does), so each
	// GPU write is checked once. A texture qualifies when its image can already be a color
	// attachment, as a render target the game fast-clears and then samples without drawing to it
	// first: on the CPU path each such lookup drained the GPU (once a frame near Astro Bot's
	// crashed ship, waiting about 25 ms behind the frame's heaviest draw).
	const auto& view = desc.view_info;
	if ((desc.type != BindingType::RenderTarget && desc.type != BindingType::Texture) ||
	    desc.info.IsVolume() ||
	    count > DccClearResolver::MaxSlices || view.base_level != 0 ||
	    !(m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
	      vk::FormatFeatureFlagBits::eColorAttachment)) {
		return false;
	}
	{
		std::scoped_lock lock {m_lock};
		const auto&      image = m_slot_images[id];
		if (image.depth_id || image.backing.image == nullptr ||
		    !(image.backing.usage & vk::ImageUsageFlagBits::eColorAttachment) ||
		    image_first >= image.backing.layers || count > image.backing.layers - image_first) {
			return false;
		}
	}
	m_dcc_gpu_checks++;
	{
		DrainStats::ReasonScope reason(DrainStats::Reason::DccClear);
		DrainStats::Record(DrainStats::Kind::DccGpuCheck, count);
	}
	// The check consumes cleared slices on the GPU, so the metadata stays GPU-owned.
	const auto [metadata, metadata_offset] =
	    m_buffer_cache.ObtainBuffer(slices_address, slice_size * count, true, false);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	m_dcc_resolver.Record(native, *metadata, metadata_offset, slice_size, count, code_mask, true);
	{
		std::scoped_lock lock {m_lock};
		auto&            image = m_slot_images[id];
		TrackImage(id);
		if (image.IsBufferModified() || image.IsCpuDirty()) {
			// A predicated clear may not happen, so the image must already hold guest contents.
			InitializeImage(id);
			if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
				EXIT("TextureCache: DCC clear target retained guest ownership\n");
			}
		}
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentRead |
		                  vk::AccessFlagBits2::eColorAttachmentWrite,
		              {}, native);
		ImageViewInfo attachment_view {};
		attachment_view.format      = view.format;
		attachment_view.type        = count == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		attachment_view.base_layer  = image_first;
		attachment_view.layer_count = count;
		attachment_view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(attachment_view);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		const vk::Extent2D extent {image.info.extent.width, image.info.extent.height};
		vk::RenderingInfo  rendering {};
		rendering.renderArea.extent    = extent;
		rendering.layerCount           = count;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		native.beginRendering(&rendering);
		for (uint32_t slice = 0; slice < count; slice++) {
			for (uint32_t index = 0; index < DccClearResolver::CodeCount; index++) {
				if ((code_mask & (1u << index)) == 0) {
					continue;
				}
				vk::ConditionalRenderingBeginInfoEXT condition {};
				condition.buffer = m_dcc_resolver.PredicateBuffer();
				condition.offset = DccClearResolver::PredicateOffset(slice, index);
				native.beginConditionalRenderingEXT(&condition);
				const vk::ClearAttachment clear {vk::ImageAspectFlagBits::eColor, 0,
				                                 vk::ClearValue {colors[index]}};
				const vk::ClearRect       rect {vk::Rect2D {{0, 0}, extent}, slice, 1};
				native.clearAttachments(1, &clear, 1, &rect);
				native.endConditionalRenderingEXT();
			}
		}
		native.endRendering();
		// Transit does not separate attachment writes in the same layout; order the clears
		// before the draw that follows.
		vk::MemoryBarrier2 ordered {};
		ordered.srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
		ordered.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
		ordered.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		ordered.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &ordered;
		native.pipelineBarrier2(dependency);
		CommitGpuWrite(image);
	}
	MarkDccSlicesChecked(slices_address, slice_size, count, code_mask);
	return true;
}

bool TextureCache::DccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count,
                                    uint32_t code_mask) {
	std::scoped_lock lock {m_dcc_checked_mutex};
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto found = m_dcc_checked.find(address + slice_size * slice);
		if (found == m_dcc_checked.end() || found->second.size != slice_size ||
		    (found->second.code_mask & code_mask) != code_mask) {
			return false;
		}
	}
	return true;
}

void TextureCache::MarkDccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count,
                                        uint32_t code_mask) {
	std::scoped_lock lock {m_dcc_checked_mutex};
	for (uint32_t slice = 0; slice < count; slice++) {
		m_dcc_checked[address + slice_size * slice] = {slice_size, code_mask};
	}
	m_dcc_checked_generation.fetch_add(1, std::memory_order_release);
}

void TextureCache::OnBufferGpuWrite(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_dcc_checked_mutex};
	if (m_dcc_checked.empty() || size == 0) {
		return;
	}
	auto entry = m_dcc_checked.lower_bound(address);
	if (entry != m_dcc_checked.begin()) {
		const auto previous = std::prev(entry);
		if (previous->first + previous->second.size > address) {
			entry = previous;
		}
	}
	while (entry != m_dcc_checked.end() && entry->first < address + size) {
		entry = m_dcc_checked.erase(entry);
		m_dcc_checked_generation.fetch_add(1, std::memory_order_release);
	}
}

void TextureCache::RefreshImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		const auto hash = image.HashGuestEdges();
		if (image.NeedsMaybeCpuHash()) {
			image.SetMaybeCpuHash(hash);
			return;
		}
		(void)image.ResolveMaybeCpuHash(hash);
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	if (GpuZones::Enabled()) [[unlikely]] {
		LogZoneRefresh(image);
	}
	InitializeImage(id);
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id || record.stencil_subresources != depth.stencil_subresources) {
		ImageInfo plane {};
		plane.data         = stencil;
		plane.pixel_format = vk::Format::eS8Uint;
		plane.guest_format = Prospero::BufferFormat::k8UInt;
		plane.type         = depth.info.type;
		plane.extent       = depth.info.extent;
		plane.resources    = {depth.stencil_subresources.level_count,
		                      depth.stencil_subresources.layer_count};
		plane.tile_mode    = depth.info.tile_mode;
		plane.pitch =
		    plane.IsTiled() ? TileGetDepthPitch(plane.extent.width, 1, 0) : depth.info.pitch;
		plane.bytes_per_block = 1;
		plane.UpdateSize();
		record.info                 = plane;
		record.depth_id             = depth_id;
		record.stencil_subresources = depth.stencil_subresources;
	}

	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format, uint64_t* unique_generation) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	if (unique_generation != nullptr) {
		*unique_generation = 0;
	}
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::unique_lock lock {m_lock};
		int32_t view_mip        = -1;
		int32_t view_layer      = -1;
		int32_t backing_matches = 0;
		// The images over the request's range, and the first of them.
		size_t  candidate_count = 0;
		ImageId first_candidate {};
		// Vulkan renders no color into a stencil aspect: a color target over a depth image's
		// stencil plane (PPSA34015 draws R8 targets there) gets an image of its own.
		const bool color_target =
		    desc.type == BindingType::RenderTarget || desc.type == BindingType::VideoOut;
		const auto stencil_plane = [&](ImageId id) {
			return color_target && m_slot_images[id].depth_id;
		};
		const auto       lookup     = [&] {
			result          = {};
			view_mip        = -1;
			view_layer      = -1;
			backing_matches = 0;
			const auto candidates =
			    FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
			candidate_count = candidates.size();
			first_candidate = candidates.empty() ? ImageId {} : candidates.front();
			for (const auto id: candidates) {
				const auto& image = m_slot_images[id];
				if (stencil_plane(id)) {
					continue;
				}
				// Stencil descriptors name the associated plane; its depth owner supplies
				// the native image and the canonical guest tiling. Not a backing match: the
				// rule depends on the binding type, which ImageLookup does not key.
				if ((desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
				    image.depth_id && image.info.data == desc.info.data &&
				    m_slot_images[image.depth_id].info.samples == desc.info.samples &&
				    image.info.extent == desc.info.extent && image.info.type == desc.info.type &&
				    image.info.resources.levels >= desc.info.resources.levels &&
				    image.info.resources.layers >= desc.info.resources.layers &&
				    desc.info.bytes_per_block == 1 &&
				    desc.info.guest_format == Prospero::BufferFormat::k8UInt) {
					result          = id;
					backing_matches = 0;
					break;
				}
				if (SameBacking(image.info, desc.info, exact_format)) {
					result = id;
					backing_matches++;
				}
			}
			if (!result) {
				for (const auto candidate: candidates) {
					if (stencil_plane(candidate)) {
						continue;
					}
					view_mip                = -1;
					view_layer              = -1;
					const auto& merged_info = result ? m_slot_images[result].info : desc.info;
					const auto  overlap = ResolveOverlap(merged_info, desc.type, candidate, result);
					if (overlap.image) {
						result     = overlap.image;
						view_mip   = overlap.mip;
						view_layer = overlap.layer;
					}
				}
			}
			if (result && !m_slot_images[result].depth_id) {
				auto& resolved = m_slot_images[result];
				if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
					result = {};
				} else if (resolved.info.resources < desc.info.resources) {
					ChurnReason reason("find-grow");
					FreeImage(result);
					result = {};
				}
			}
		};
		// Draws look up the same textures again and again; see ImageLookup.
		static const bool memo_enabled = [] {
			const char* text = std::getenv("KYTY_DEBUG_IMAGE_MEMO");
			return text == nullptr || std::strcmp(text, "0") != 0;
		}();
		auto* const  lookups          = ImageLookupSet(desc.info, exact_format);
		const auto   range_generation = RangeGeneration(desc.info.data);
		ImageLookup* remembered       = nullptr;
		for (size_t way = 0; memo_enabled && way < ImageLookupWays; way++) {
			if (lookups[way].generation == range_generation &&
			    lookups[way].Matches(desc.info, exact_format) && !stencil_plane(lookups[way].id)) {
				remembered = &lookups[way];
				break;
			}
		}
		// KYTY_VERIFY_IMAGE_MEMO=1: a remembered lookup is looked up again and must agree.
		static const bool verify_memo = std::getenv("KYTY_VERIFY_IMAGE_MEMO") != nullptr;
		if (remembered != nullptr && verify_memo) [[unlikely]] {
			const auto remembered_id = remembered->id;
			lookup();
			const bool same = result == remembered_id && backing_matches == 1 && view_mip < 0 &&
			                  view_layer < 0;
			static std::atomic<uint64_t> checked {0};
			static std::atomic<uint64_t> missed {0};
			const auto count = checked.fetch_add(1, std::memory_order_relaxed) + 1;
			if (!same && missed.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::printf("image-memo verify: 0x%016" PRIx64 " size 0x%" PRIx64
				            " remembered a different lookup (matches %d, mip %d, layer %d)\n",
				            desc.info.data.address, desc.info.data.size, backing_matches, view_mip,
				            view_layer);
			}
			if (count % 100000 == 0) {
				std::printf("image-memo verify: hits=%" PRIu64 " missed=%" PRIu64 "\n", count,
				            missed.load(std::memory_order_relaxed));
				std::fflush(stdout);
			}
			// Go on with the lookup's own result, as without the memo.
			remembered = nullptr;
			if (!same) {
				result = {};
			}
		}
		// A request that one image holds as a view: see OverlapLookup. KYTY_DEBUG_AB=overlapmemo
		// resolves every such overlap in alternate windows.
		static const bool overlap_ab      = AbSelected("overlapmemo");
		const bool        overlap_enabled = memo_enabled && !(overlap_ab && AbFeatureOff());
		auto&             overlap         = OverlapLookupFor(desc.info);
		bool overlap_hit = remembered == nullptr && overlap_enabled &&
		                   overlap.generation == range_generation && overlap.binding == desc.type &&
		                   overlap.exact_format == exact_format &&
		                   SameImageInfo(overlap.info, desc.info);
		if (overlap_hit && verify_memo) [[unlikely]] {
			// KYTY_VERIFY_IMAGE_MEMO=1: the overlap is resolved again and must give the same image
			// without creating or freeing one.
			const auto remembered_id = overlap.id;
			lookup();
			const bool same = result == remembered_id && backing_matches == 0 && view_mip < 0 &&
			                  view_layer < 0 && RangeGeneration(desc.info.data) == range_generation;
			static std::atomic<uint64_t> checked {0};
			static std::atomic<uint64_t> missed {0};
			const auto count = checked.fetch_add(1, std::memory_order_relaxed) + 1;
			if (!same && missed.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::printf("image-memo verify: 0x%016" PRIx64 " size 0x%" PRIx64
				            " remembered a different overlap (matches %d, mip %d, layer %d)\n",
				            desc.info.data.address, desc.info.data.size, backing_matches, view_mip,
				            view_layer);
			}
			if (count % 100000 == 0) {
				std::printf("image-memo verify: overlap hits=%" PRIu64 " missed=%" PRIu64 "\n", count,
				            missed.load(std::memory_order_relaxed));
				std::fflush(stdout);
			}
			// Go on with the lookup's own result, as without the memo.
			overlap_hit = false;
			if (!same) {
				result = {};
			}
		}
		if (remembered != nullptr) {
			remembered->last_use = ++m_image_lookup_clock;
			result               = remembered->id;
			if (unique_generation != nullptr) {
				*unique_generation = range_generation;
			}
		} else if (overlap_hit) {
			result = overlap.id;
			if (unique_generation != nullptr) {
				*unique_generation = range_generation;
			}
		} else {
			lookup();
			if (!result && m_last_pressure_gc_tick != m_gc_tick &&
			    m_graphics.CanReportMemoryUsage() &&
			    m_graphics.GetDeviceMemoryUsage() >= m_pressure_gc_memory) {
				// At most once per submission age: image misses must not age live resources or
				// repeatedly drain a working set that cannot be reclaimed losslessly.
				m_last_pressure_gc_tick = m_gc_tick;
				lock.unlock();
				(void)CollectGarbage(true);
				lock.lock();
				// Collection invalidates aliases and can retire the source of this lookup.
				lookup();
			}
			if (!result) {
				ChurnReason reason("find-miss");
				result         = InsertImage(desc.info);
				auto&      inserted  = m_slot_images[result];
				const bool gpu_dirty = m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
				                                                       inserted.info.data.size);
				if (gpu_dirty) {
					inserted.MarkBufferModified();
				}
				static const bool log_misses = std::getenv("KYTY_DEBUG_MISS_LOOKUPS") != nullptr;
				if (log_misses) [[unlikely]] {
					std::printf("miss-lookup: frame=%d type=%u 0x%016" PRIx64 " size=0x%" PRIx64
					            " %ux%ux%u fmt=%u tile=%u mips=%u gpu_dirty=%d tick=%" PRIu64 "\n",
					            m_scheduler.Context().GetGpu().GetFrameNum(),
					            static_cast<uint32_t>(desc.type), inserted.info.data.address,
					            inserted.info.data.size, inserted.info.extent.width,
					            inserted.info.extent.height, inserted.info.resources.layers,
					            static_cast<uint32_t>(inserted.info.pixel_format),
					            static_cast<uint32_t>(inserted.info.tile_mode),
					            inserted.info.resources.levels, gpu_dirty ? 1 : 0,
					            m_scheduler.CurrentTick());
				}
			} else if (backing_matches == 1 && view_mip < 0 && view_layer < 0) {
				// The lookup can have registered or freed images: take the range's generation now.
				const auto found_generation = RangeGeneration(desc.info.data);
				auto*      victim           = lookups;
				for (size_t way = 0; way < ImageLookupWays; way++) {
					if (lookups[way].generation != RangeGeneration(lookups[way].data)) {
						victim = &lookups[way];
						break;
					}
					if (lookups[way].last_use < victim->last_use) {
						victim = &lookups[way];
					}
				}
				*victim = {.generation      = found_generation,
				           .last_use        = ++m_image_lookup_clock,
				           .data            = desc.info.data,
				           .extent          = desc.info.extent,
				           .resources       = desc.info.resources,
				           .samples         = desc.info.samples,
				           .bytes_per_block = desc.info.bytes_per_block,
				           .tile_mode       = desc.info.tile_mode,
				           .pixel_format    = desc.info.pixel_format,
				           .type            = desc.info.type,
				           .exact_format    = exact_format,
				           .id              = result};
				if (unique_generation != nullptr) {
					*unique_generation = found_generation;
				}
			} else if (overlap_enabled && backing_matches == 0 && candidate_count == 1 &&
			           result == first_candidate && view_mip < 0 && view_layer < 0 &&
			           RangeGeneration(desc.info.data) == range_generation) {
				// The only image over the range holds the request as a view, and resolving that
				// neither registered nor freed an image (the generation would have advanced).
				overlap = {.generation   = range_generation,
				           .info         = desc.info,
				           .binding      = desc.type,
				           .exact_format = exact_format,
				           .id           = result};
				if (unique_generation != nullptr) {
					*unique_generation = range_generation;
				}
			}
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		static const bool log_mismatch = std::getenv("KYTY_DEBUG_MISS_LOOKUPS") != nullptr;
		if (log_mismatch && view_mip < 0 && view_layer < 0 &&
		    (image.info.extent.width != desc.info.extent.width ||
		     image.info.extent.height != desc.info.extent.height)) [[unlikely]] {
			std::printf("mismatch-lookup: frame=%d type=%u req=0x%016" PRIx64 " %ux%u fmt=%u "
			            "tile=%u -> 0x%016" PRIx64 " %ux%u fmt=%u tile=%u gpu=%d\n",
			            m_scheduler.Context().GetGpu().GetFrameNum(),
			            static_cast<uint32_t>(desc.type), desc.info.data.address,
			            desc.info.extent.width, desc.info.extent.height,
			            static_cast<uint32_t>(desc.info.pixel_format),
			            static_cast<uint32_t>(desc.info.tile_mode), image.info.data.address,
			            image.info.extent.width, image.info.extent.height,
			            static_cast<uint32_t>(image.info.pixel_format),
			            static_cast<uint32_t>(image.info.tile_mode), image.IsGpuModified() ? 1 : 0);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
		static const bool log_depth = std::getenv("KYTY_DEBUG_DEPTH_LOOKUPS") != nullptr;
		static std::unordered_map<uint64_t, uint64_t> last_depth_result;
		const uint64_t depth_key = desc.info.data.address ^
		                           (uint64_t {desc.info.extent.width} << 48u) ^
		                           (uint64_t {desc.info.extent.height} << 32u) ^
		                           (uint64_t {static_cast<uint32_t>(desc.type)} << 62u);
		const uint64_t depth_value =
		    (uint64_t {result.index} << 32u) | result.generation;
		if (log_depth && (desc.info.IsDepth() || image.info.IsDepth()) &&
		    std::exchange(last_depth_result[depth_key], depth_value) != depth_value) [[unlikely]] {
			std::printf("depth-lookup: frame=%d type=%u req=0x%016" PRIx64 " %ux%u fmt=%u -> "
			            "img=%u 0x%016" PRIx64 " %ux%u fmt=%u gpu=%d mip=%d layer=%d tick=%" PRIu64
			            "\n",
			            m_scheduler.Context().GetGpu().GetFrameNum(),
			            static_cast<uint32_t>(desc.type), desc.info.data.address,
			            desc.info.extent.width, desc.info.extent.height,
			            static_cast<uint32_t>(desc.info.pixel_format), result.index,
			            image.info.data.address, image.info.extent.width, image.info.extent.height,
			            static_cast<uint32_t>(image.info.pixel_format), image.IsGpuModified() ? 1 : 0,
			            static_cast<int>(desc.view_info.base_level),
			            static_cast<int>(desc.view_info.base_layer), m_scheduler.CurrentTick());
		}
	}
	FinishFind(result, desc, metadata_base_layer);
	return result;
}

bool TextureCache::RefindImage(ImageId id, uint64_t generation, const ImageDesc& desc,
                               uint32_t metadata_base_layer) {
	// Only Thread_Gpu (the caller) registers, frees and touches images, so when this tick and GC
	// tick already saw the image, the bookkeeping below would change nothing: skip the lock.
	if (generation == 0 || generation != RangeGeneration(desc.info.data)) {
		return false;
	}
	const auto tick  = m_scheduler.CurrentTick();
	auto&      image = m_slot_images[id];
	if (image.tick_accessed_last != tick || (image.registered && image.lru_tick != m_gc_tick)) {
		std::scoped_lock lock {m_lock};
		if (generation != RangeGeneration(desc.info.data)) {
			return false;
		}
		image.tick_accessed_last = tick;
		TouchImage(image);
	}
	FinishFind(id, desc, metadata_base_layer);
	return true;
}

bool TextureCache::IsTextureCurrent(ImageId id, uint64_t generation) const noexcept {
	// The image may have been freed since; a slot reused by another image has a newer generation.
	const auto* owner = m_slot_images.try_get(id);
	if (owner == nullptr || generation == 0 || generation != RangeGeneration(owner->info.data)) {
		return false;
	}
	return IsTextureClean(id);
}

bool TextureCache::IsTextureClean(ImageId id) const noexcept {
	const auto& image = m_slot_images[id];
	// FindTexture's RefreshImage and stencil refresh would do nothing: the image is tracked over
	// its whole range and neither CPU nor buffer writes made it dirty.
	return image.registered && !image.depth_id && !image.binding.needs_rebind &&
	       !image.info.data.Empty() && !image.info.HasStencil() && !image.IsCpuDirty() &&
	       !image.IsBufferModified() && image.track_addr == image.info.data.address &&
	       image.track_addr_end == image.info.data.End();
}

bool TextureCache::IsRenderTargetCurrent(ImageId id, uint64_t generation) const noexcept {
	if (!IsTextureCurrent(id, generation)) {
		return false;
	}
	// FindRenderTarget's MarkGpuModified, CommitGpuWrite and usage flag would change nothing, and
	// TrackImageDownload enrolls nothing (linear readback images re-enroll every acquisition).
	const auto& image = m_slot_images[id];
	return image.IsGpuModified() && image.usage.render_target && image.backing.image != nullptr &&
	       !(m_readback_linear_images && !image.info.IsTiled());
}

bool TextureCache::IsStorageCurrent(ImageId id, uint64_t generation) const noexcept {
	if (!IsTextureCurrent(id, generation)) {
		return false;
	}
	// FindTexture's MarkGpuModified and CommitGpuWrite would change nothing, and TrackImageDownload
	// enrolls nothing (linear readback images re-enroll every acquisition).
	const auto& image = m_slot_images[id];
	return image.IsGpuModified() && image.backing.image != nullptr &&
	       !(m_readback_linear_images && !image.info.IsTiled());
}

bool TextureCache::IsDepthTargetCurrent(ImageId id, uint64_t generation, uint64_t meta_generation,
                                        const ImageDesc& desc, ImageId stencil,
                                        uint64_t stencil_generation) const noexcept {
	// KYTY_DEBUG_AB=stencilreuse acquires targets with stencil every draw in alternate windows.
	static const bool stencil_ab = AbSelected("stencilreuse");
	if (meta_generation != m_surface_meta_generation.load(std::memory_order_acquire) ||
	    (desc.info.HasStencil() && stencil_ab && AbFeatureOff())) {
		return false;
	}
	// As IsTextureCurrent, for an image that may have a stencil plane: it and its views still
	// live, and it is tracked over its whole range and neither CPU nor buffer writes made it dirty.
	const auto* owner = m_slot_images.try_get(id);
	if (owner == nullptr || generation == 0 || generation != RangeGeneration(owner->info.data)) {
		return false;
	}
	const auto& image = *owner;
	if (!image.registered || image.depth_id || image.binding.needs_rebind ||
	    image.info.data.Empty() || image.IsCpuDirty() || image.IsBufferModified() ||
	    image.track_addr != image.info.data.address ||
	    image.track_addr_end != image.info.data.End()) {
		return false;
	}
	// FindDepthTarget's MarkGpuModified, CommitGpuWrite, usage flag and stencil and metadata
	// assignments would change nothing, and the metadata entry it made is still there.
	if (!image.IsGpuModified() || !image.usage.depth_target || image.backing.image == nullptr ||
	    !(image.info.stencil == desc.info.stencil) || !(image.info.metadata == desc.info.metadata)) {
		return false;
	}
	if (!desc.info.HasStencil()) {
		return true;
	}
	// AssociateStencil would find this record again (the same images are over the plane), touch it
	// to no effect (it is this GC tick's) and leave it this image's; RefreshImage would find it
	// tracked and clean.
	const auto* record = m_slot_images.try_get(stencil);
	return record != nullptr && stencil_generation != 0 &&
	       stencil_generation == RangeGeneration(desc.info.stencil) && record->registered &&
	       record->depth_id == id && record->lru_tick == m_gc_tick &&
	       record->info.data == desc.info.stencil && record->info.extent == image.info.extent &&
	       !record->IsCpuDirty() && !record->IsBufferModified() &&
	       record->track_addr == record->info.data.address &&
	       record->track_addr_end == record->info.data.End();
}

// The part of FindImage after the lookup that runs without the lock.
void TextureCache::FinishFind(ImageId id, const ImageDesc& desc, uint32_t metadata_base_layer) {
	DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::FindFinish);
	MaterializeColorClear(id, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[id];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	ImageIds         matches;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = m_slot_images.try_get(owner->depth_id);
		}
		if (owner == nullptr || (ensure_valid && !owner->SafeToDownload())) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid) {
		const auto owner = m_slot_images.try_get(selected);
		if (owner != nullptr && owner->depth_id) {
			selected = owner->depth_id;
		}
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	RefreshVia       via(desc.type == BindingType::Storage ? "storage" : "texture");
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	if (desc.type == BindingType::Storage) {
		MarkGpuModified(image);
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			for (const auto stencil_id:
			     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
				const auto* stencil = m_slot_images.try_get(stencil_id);
				if (stencil != nullptr && stencil->depth_id == id &&
				    stencil->info.data == image.info.stencil) {
					RefreshImage(stencil_id);
					break;
				}
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
			}
			TrackImageDownload(id, image);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	RefreshVia       via("render-target");
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition: registered=%d depth_alias=%d needs_rebind=%d addr=0x%016" PRIx64 " size=0x%" PRIx64 " %ux%u\n", static_cast<int>(image.registered), static_cast<int>(static_cast<bool>(image.depth_id)), static_cast<int>(image.binding.needs_rebind), static_cast<uint64_t>(image.info.data.address), static_cast<uint64_t>(image.info.data.size), image.info.extent.width, image.info.extent.height);
	}
	TouchImage(image);
	MarkGpuModified(image);
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc, ImageId* stencil) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	if (stencil != nullptr) {
		*stencil = {};
	}
	RefreshVia       via("depth-target");
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	MarkGpuModified(image);
	image.usage.depth_target = true;
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	// The overlap rules read whether an image has a stencil plane and metadata (see OverlapLookup).
	if (image.info.HasStencil() != desc.info.HasStencil() ||
	    image.info.HasMetadata() != desc.info.HasMetadata()) {
		AdvanceImageSetGeneration(image.info.data);
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		const auto& range = desc.info.metadata.range;
		// Rendered contents are not a clear: an image whose entry went away (an aliasing image
		// freed it) only starts cleared again through a recognized clear of the HTile.
		const auto [meta, inserted] = m_surface_metas.emplace(
		    range.address,
		    MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                  .clear_mask = image.IsGpuModified() ? 0 : image.info.htile_clear_mask});
		if (inserted) {
			const auto tick = m_scheduler.CurrentTick();
			const auto fill = std::ranges::find_if(m_unclaimed_fills, [&](const UnclaimedFill& f) {
				return f.address == range.address && f.size >= range.size &&
				       tick - std::min(tick, f.tick) <= UnclaimedFillTicks;
			});
			if (fill != m_unclaimed_fills.end()) {
				meta->second.clear_mask = UINT32_MAX;
				m_meta_clear_generation.fetch_add(1, std::memory_order_release);
				m_unclaimed_fills.erase(fill);
				if (DebugFrameDrawsEnabled()) [[unlikely]] {
					std::printf("frame-draws: frame=%d htile-claim 0x%016" PRIx64 " size=0x%" PRIx64
					            "\n",
					            m_scheduler.Context().GetGpu().GetFrameNum(), range.address,
					            range.size);
				}
			} else if (DebugFrameDrawsEnabled()) [[unlikely]] {
				std::printf("frame-draws: frame=%d htile-unclaimed 0x%016" PRIx64 " size=0x%" PRIx64
				            " depth=0x%016" PRIx64 " %ux%u mask=%x fills:",
				            m_scheduler.Context().GetGpu().GetFrameNum(), range.address, range.size,
				            desc.info.data.address, desc.info.extent.width,
				            desc.info.extent.height, image.info.htile_clear_mask);
				for (const auto& f: m_unclaimed_fills) {
					std::printf(" 0x%" PRIx64 "+0x%" PRIx64 "@%" PRIu64, f.address, f.size,
					            tick - std::min(tick, f.tick));
				}
				std::printf("\n");
			}
			m_surface_meta_generation.fetch_add(1, std::memory_order_relaxed);
		}
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		const auto association = AssociateStencil(id, desc.info.stencil);
		RefreshImage(association);
		if (stencil != nullptr) {
			*stencil = association;
		}
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	MarkGpuModified(image);
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner->info.resources.levels, 0, owner->info.TransferLayers()};
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner->depth_id;
			candidate_subresources = owner->stencil_subresources;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		ImageViewInfo view {};
		view.format = format;
		view.type   = range.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		view.base_level  = range.baseMipLevel;
		view.base_layer  = range.baseArrayLayer;
		view.layer_count = range.layerCount;
		view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, command.Handle());
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(view);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eClear;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		attachment.clearValue  = clear;
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = {
		    std::max(image.info.extent.width >> range.baseMipLevel, 1u),
		    std::max(image.info.extent.height >> range.baseMipLevel, 1u)};
		rendering.layerCount           = range.layerCount;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		command.Handle().beginRendering(&rendering);
		command.Handle().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Handle());
	auto native_range = range;
	GpuZones::Mark(command.Handle(), DrainStats::Zone::ImageCopy);
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Handle().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported image invalidation from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     address, size);
	}
	// Invalidation can release a page's write watcher even when the fault is just outside
	// an image's byte range. Finish every writeback on that page before permitting CPU stores.
	const auto page_begin = std::max<uint64_t>(Common::AlignDown(address, TRACKER_PAGE_SIZE), 1);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	const auto page_size  = page_end - page_begin;
	std::unique_lock lock {m_lock};
	while (HasPendingDownload(page_begin, page_size)) {
		// The GPU thread can need m_lock before reaching this command. Never wait for that
		// thread while holding the texture lock on a CPU fault path.
		lock.unlock();
		m_scheduler.Context().GetGpu().SendCommandSync([this, page_begin, page_size] {
			DrainStats::ReasonScope reason(DrainStats::Reason::TexturePendingDownload);
			if (HasPendingDownload(page_begin, page_size)) {
				const auto tick = m_scheduler.CurrentTick();
				m_scheduler.Wait(tick);
				m_scheduler.WaitPriorityOperations(tick);
			}
		});
		lock.lock();
		// A new download may have been recorded before we reacquired m_lock. Its publication
		// must also complete; checking and invalidating under this lock excludes another one.
	}
	InvalidateCpuAliases(address, size);
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.linear_slice_stride;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			const auto copies = info.BufferCopies(destination_offset);
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
		} else {
			const auto copies = info.BufferCopies();
			m_tiler.TileImage(image, copies, destination.Handle(), destination_offset,
			                  info.data.size);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	const std::array copies {vk::BufferImageCopy {0,
	                                              info.pitch,
	                                              info.extent.height,
	                                              {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                                              {},
	                                              info.extent}};
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto guest_linear = tiled ? m_tiler.GetScratchBuffer(info.LinearSize(), host_linear.buffer)
	                          : TileManager::Result {destination.Handle(), destination_offset,
	                                                 destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (tiled) {
		m_tiler.Tile(guest_linear.buffer, guest_linear.offset, guest_linear.size,
		             destination.Handle(), destination_offset, info.data.size, info);
	}
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                 uint64_t destination_size, uint32_t levels) {
	GpuZones::KeyScope zone_key(image.info.data.address | GpuZones::DownloadKey);
	if (!CanDownload(image)) {
		EXIT("TextureCache: invalid image download\n");
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		EXIT_IF(destination_size != image.info.data.size || levels != image.info.resources.levels);
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	const auto& info      = image.info;
	const auto  transform = info.GetColorTransform();
	if (info.IsTiled()) {
		const auto copies = info.BufferCopies(0, levels);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, destination_size,
		                  levels);
	} else if (transform != ColorTransform::None) {
		auto       linear = m_tiler.GetScratchBuffer(destination_size);
		const auto copies = info.BufferCopies(0, levels);
		image.Download(copies, linear.buffer, 0, linear.size);
		m_tiler.TransformColor(linear, {destination.Handle(), destination_offset, destination_size},
		                       transform, false);
	} else {
		const auto copies = info.BufferCopies(destination_offset, levels);
		image.Download(copies, destination.Handle(), destination_offset, destination_size);
	}
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain.
		return false;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	if (!m_texture_cache.CanDownload(image) ||
	    (TextureCache::UploadBinding(image) == TextureCache::BindingType::DepthTarget &&
	     copy_size != image.info.data.size)) {
		return false;
	}
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, levels);
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	if (!image.SafeToDownload() || !CanDownload(image)) {
		return false;
	}
	const auto range    = image.info.data;
	auto&      staging  = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	auto [mapped, offset] =
	    staging.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	std::unique_ptr<Buffer> temporary;
	Buffer*                 download = &staging;
	if (mapped == nullptr) {
		// Tiled and format-converted downloads also write via compute shaders, so the fallback
		// must retain storage-buffer usage rather than only supporting transfer destinations.
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     AllFlags, range.size);
		download  = temporary.get();
		mapped    = download->Mapped().data();
		offset    = 0;
		EXIT_IF(mapped == nullptr);
	} else {
		staging.Commit();
	}
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	download->Flush(offset, range.size);

	DownloadImage(image, *download, offset, range.size, image.info.resources.levels);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	{
		std::lock_guard lock(m_pending_download_mutex);
		m_pending_downloads.push_back(range);
	}
	m_scheduler.DeferPriorityOperation(
	    [this, download, owner = std::move(temporary), range, mapped, offset] {
		    download->Invalidate(offset, range.size);
		    LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
		    // Separate from m_lock: the GPU thread may wait for this callback while it owns the
		    // texture cache lock (for example on staging wrap). Keep overlapping records distinct.
		    std::lock_guard lock(m_pending_download_mutex);
		    const auto      pending = std::ranges::find(m_pending_downloads, range);
		    EXIT_IF(pending == m_pending_downloads.end());
		    m_pending_downloads.erase(pending);
		    (void)owner;
	    });
	return true;
}

bool TextureCache::HasPendingDownload(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::lock_guard lock(m_pending_download_mutex);
	return std::ranges::any_of(m_pending_downloads, [address, size](const GuestRange& pending) {
		return pending.address < address + size && address < pending.End();
	});
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size,
                                           GpuWriteSource source) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	const auto images = FindImagesInRegion(address, size, true);
	// A shader writing exactly one image's range as a buffer (Sky Garden writes several tiled
	// render-target surfaces so, each a few times a frame) changes that image and those inside
	// the range. An image the range only partly overlaps is another surface using the memory at
	// another time: on the console the bytes outside the range keep its contents, and the bytes
	// inside are another surface's layout, which it does not read. Re-uploading it from the
	// buffer instead lost the rest of its contents and cost a full-size detile per write.
	// KYTY_DEBUG_AB=exactwrite marks every overlapped image again in every other window.
	static const bool ab    = AbSelected("exactwrite");
	bool              exact = false;
	if (!(ab && AbFeatureOff())) {
		for (const auto id: images) {
			const auto& data = m_slot_images[id].info.data;
			exact            = exact || (data.address == address && data.size == size);
		}
	}
	for (const auto id: images) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (exact && (image.info.data.address < address ||
		              image.info.data.End() > address + size)) {
			continue;
		}
		if (GpuZones::Enabled()) [[unlikely]] {
			LogZoneBufferWrite(image, address, size, source);
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
}

bool TextureCache::IsRegionGpuModified(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	// Strict SRT reads ask this for the same small CPU-written tables every draw. A tracker page
	// without a GPU-modified image stays so until an image is registered or becomes GPU-modified;
	// every other change only cleans pages further (and stencil associations never go away).
	struct CleanPage {
		uint64_t page       = UINT64_MAX;
		uint64_t images     = 0;
		uint64_t gpu_writes = 0;
	};
	thread_local std::array<CleanPage, 64> clean_pages {};
	// KYTY_DEBUG_AB=cleanpages scans the images every time in every other window.
	static const bool ab       = AbSelected("cleanpages");
	const auto        page     = address / TRACKER_PAGE_SIZE;
	const bool        one_page = (address + size - 1) / TRACKER_PAGE_SIZE == page &&
	                      !(ab && AbFeatureOff());
	auto& slot = clean_pages[page % clean_pages.size()];
	if (one_page && slot.page == page &&
	    slot.images == m_image_set_generation.load(std::memory_order_acquire) &&
	    slot.gpu_writes == m_gpu_modified_generation.load(std::memory_order_acquire)) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	// Taken before the scan, so a change racing with it leaves the record stale, never clean.
	const auto images       = m_image_set_generation.load(std::memory_order_acquire);
	const auto gpu_writes   = m_gpu_modified_generation.load(std::memory_order_acquire);
	// PPSA17168: S_LOAD_DWORD reads shader data at an address overlapping an old render target
	// whose memory the CPU has reused. The cached image still retains its earlier GPU-modified
	// flag, so a definitely CPU-dirty image doesn't count. The clean-page record ignores that
	// exception: CPU-dirty clears without a generation bump, so only pages without any
	// GPU-modified image are recorded.
	const auto gpu_modified = [this](uint64_t begin, uint64_t bytes, bool honor_cpu_dirty) {
		for (const auto id: FindImagesInRegion(begin, bytes, false)) {
			const auto& image = m_slot_images[id];
			if (!image.depth_id && image.IsGpuModified() &&
			    !(honor_cpu_dirty && image.IsDefinitelyCpuDirty())) {
				return true;
			}
		}
		return false;
	};
	if (one_page && !gpu_modified(page * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE, false)) {
		slot = {page, images, gpu_writes};
		return false;
	}
	return gpu_modified(address, size, true);
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (owner->Overlaps(address, size)) {
			owner->InvalidateCpuWrite(address, size);
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner->info.data.address;
		const auto image_end   = owner->info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, *owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsKnownDccMetadata(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return m_dcc_metadata_seen.Intersects(address, size);
}

// Draws ask about the same depth surface over and over. The answer holds until its clear state
// changes or a surface metadata entry comes or goes, so the last one is kept without m_lock.
// KYTY_DEBUG_AB=metamemo looks every answer up in every other window.
bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	static const bool ab       = AbSelected("metamemo");
	auto&             memo     = m_meta_clear_memo;
	// Read before the lookup: a change racing with it leaves the memo stale, never wrong.
	const auto        surfaces = m_surface_meta_generation.load(std::memory_order_acquire);
	const auto        clears   = m_meta_clear_generation.load(std::memory_order_acquire);
	if (!(memo.valid && memo.address == address && memo.surface_generation == surfaces &&
	      memo.clear_generation == clears) ||
	    (ab && AbFeatureOff())) {
		std::scoped_lock lock {m_lock};
		const auto       found = m_surface_metas.find(address);
		memo = {.address            = address,
		        .surface_generation = surfaces,
		        .clear_generation   = clears,
		        .clear_mask         = found != m_surface_metas.end() ? found->second.clear_mask : 0,
		        .found              = found != m_surface_metas.end(),
		        .valid              = true};
	}
	return memo.found && slice < 32 && (memo.clear_mask & (1u << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	m_meta_clear_generation.fetch_add(1, std::memory_order_release);
	return true;
}

static bool UnclaimedFillsEnabled() {
	static const bool enabled = std::getenv("KYTY_DEBUG_NO_UNCLAIMED_FILLS") == nullptr;
	return enabled;
}

void TextureCache::NoteUnclaimedFill(uint64_t address, uint64_t size) {
	if (!UnclaimedFillsEnabled() || address == 0 || size == 0) {
		return;
	}
	std::scoped_lock lock {m_lock};
	const auto       tick = m_scheduler.CurrentTick();
	std::erase_if(m_unclaimed_fills, [&](const UnclaimedFill& fill) {
		return fill.address == address || tick - std::min(tick, fill.tick) > UnclaimedFillTicks;
	});
	if (m_unclaimed_fills.size() >= UnclaimedFillsMax) {
		m_unclaimed_fills.erase(m_unclaimed_fills.begin());
	}
	m_unclaimed_fills.push_back({.address = address, .size = size, .tick = tick});
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	const auto previous = found->second.clear_mask;
	if (is_clear) {
		found->second.clear_mask |= 1u << slice;
	} else {
		found->second.clear_mask &= ~(1u << slice);
	}
	if (found->second.clear_mask != previous) {
		m_meta_clear_generation.fetch_add(1, std::memory_order_release);
	}
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	ChurnReason reason("unmap");
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	OnBufferGpuWrite(address, size);
	std::scoped_lock lock {m_lock};
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
			m_surface_meta_generation.fetch_add(1, std::memory_order_release);
		} else {
			++metadata;
		}
	}
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
}

void TextureCache::RunGarbageCollector() {
	(void)CollectGarbage(false);
}

bool TextureCache::CollectGarbage(bool pressure_only) {
	ChurnReason reason("gc");
	std::unique_lock lock {m_lock};
	const uint64_t   tick = pressure_only ? m_gc_tick : m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < (pressure_only ? m_pressure_gc_memory : m_trigger_gc_memory)) {
		return false;
	}
	bool                 reclaimed = false;
	std::vector<ImageId> pending_retirement;
	const auto collect = [&](bool allow_aggressive) {
		bool           pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool           aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		const uint64_t minimum_age = aggressive ? 160 : pressured ? 80 : 16;
		if (pressure_only && tick < minimum_age) {
			return;
		}
		const uint64_t     age            = std::min(minimum_age, tick);
		constexpr uint64_t release_margin = 256ull * 1024 * 1024;
		const auto         release_threshold =
		    m_critical_gc_memory - std::min(m_critical_gc_memory, release_margin);
		size_t         deletions = aggressive ? 40 : pressured ? 20 : 10;
		std::vector<ImageId> candidates;
		candidates.reserve(deletions);
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() == deletions;
		});
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			--deletions;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id ||
			    std::ranges::find(pending_retirement, id) != pending_retirement.end()) {
				continue;
			}
			if (pressure_only && owner->tick_accessed_last == m_scheduler.CurrentTick()) {
				continue;
			}
			if (owner->info.IsDepth()) {
				bool dirty_stencil = false;
				bool known_stencil = !owner->info.HasStencil();
				m_slot_images.ForEach([&](ImageId, const Image& associated) {
					if (associated.depth_id == id) {
						known_stencil = true;
						dirty_stencil |= associated.IsGpuModified();
					}
				});
				if (dirty_stencil || !known_stencil) {
					continue;
				}
			}
			const bool gpu_modified = owner->IsGpuModified();
			if (gpu_modified) {
				// The depth transfer preserves its plane only; the stencil ownership check above
				// prevents retiring a dirty or unknown companion. Metadata remains conservative.
				if (!pressured || !owner->SafeToDownload() || owner->info.HasMetadata() ||
				    (owner->info.IsTiled() && !aggressive)) {
					continue;
				}
				if (!DownloadImageMemory(id)) {
					continue;
				}
			}
			if (pressure_only || gpu_modified ||
			    HasPendingDownload(owner->info.data.address, owner->info.data.size)) {
				// Unregistering before publication would let CPU stores race WriteBacking.
				pending_retirement.push_back(id);
			} else {
				FreeImage(id);
			}
			reclaimed = true;
			if (m_total_used_memory < release_threshold && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
	if (!pending_retirement.empty()) {
		// Dirty eviction requires one batched completion before relinquishing guest ownership.
		// Keep m_lock so a faulting CPU writer cannot invalidate/unprotect a victim until its
		// priority writeback completes. Those callbacks use only the separate pending mutex.
		// General callbacks can replace buffers after PrepareBda while a draw is being assembled;
		// leave them for the GPU operation boundary and release only this collector's victims.
		DrainStats::ReasonScope reason(DrainStats::Reason::TextureGc);
		const auto              completion_tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(completion_tick);
		m_scheduler.WaitPriorityOperations(completion_tick);
		std::vector<ImageId> retired;
		for (const auto id: pending_retirement) {
			FreeImage(id, &retired);
		}
		for (const auto id: retired) {
			m_slot_images.erase(id);
		}
	}
	return reclaimed;
}

void TextureCache::ProcessDownloadImages() {
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			(void)DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
