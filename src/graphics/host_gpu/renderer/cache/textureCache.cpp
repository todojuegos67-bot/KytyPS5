#include "graphics/drawDiff.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/addressBindingReport.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;

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

[[nodiscard]] std::vector<vk::BufferImageCopy> BuildDepthCopies(const ImageInfo& info,
                                                              uint64_t slice_stride,
                                                              vk::ImageAspectFlags aspect) {
	std::vector<vk::BufferImageCopy> copies(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		auto& copy             = copies[layer];
		copy.bufferOffset      = slice_stride * layer;
		copy.bufferRowLength   = info.pitch;
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {aspect, 0, layer, 1};
		copy.imageExtent       = {info.extent.width, info.extent.height, 1};
	}
	return copies;
}

[[nodiscard]] std::vector<GpuTileInfo> BuildDepthTiles(const ImageInfo& info) {
	TileBlockLayout block {};
	EXIT_NOT_IMPLEMENTED(
	    !TileGetBlockLayout(TileBlockFamily::Depth64KB, info.bytes_per_block, block));
	const auto               full_slice_size = info.data.size / info.resources.layers;
	std::vector<GpuTileInfo> tiles;
	tiles.reserve(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		const auto offset = full_slice_size * layer;
		tiles.push_back({block.family, block.bytes_per_element, offset, full_slice_size, offset,
		                 full_slice_size, 0, info.extent.width, info.extent.height, 1, info.pitch});
		tiles.back().surface_z = layer;
	}
	return tiles;
}

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
}

TextureCache::~TextureCache() {
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
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

namespace {

// KYTY_IMAGE_STATS=1 (live): every 5 s, how many images were created and deleted and the guest
// ranges created most often. An image recreated every frame shows up at the top.
class ImageChurnStats {
public:
	void Note(const ImageInfo& info, bool created) {
		static auto& enabled = Common::LiveSwitches::Get("KYTY_IMAGE_STATS", 0);
		if (enabled.load(std::memory_order_relaxed) == 0) {
			return;
		}
		std::scoped_lock lock {m_mutex};
		if (created) {
			m_created++;
			m_created_bytes += info.data.size;
			const Key key {info.data.address, info.data.size, info.pixel_format, info.extent.width,
			               info.extent.height};
			m_by_range[key]++;
		} else {
			m_deleted++;
			m_deleted_bytes += info.data.size;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - m_report_time < std::chrono::seconds(5)) {
			return;
		}
		m_report_time = now;
		::printf("Images (5 s): created %" PRIu64 " (%.1f MiB), deleted %" PRIu64 " (%.1f MiB)\n",
		         m_created, static_cast<double>(m_created_bytes) / 1048576.0, m_deleted,
		         static_cast<double>(m_deleted_bytes) / 1048576.0);
		std::vector<std::pair<uint64_t, Key>> top;
		top.reserve(m_by_range.size());
		for (const auto& [key, count]: m_by_range) {
			top.emplace_back(count, key);
		}
		std::sort(top.begin(), top.end(),
		          [](const auto& a, const auto& b) { return a.first > b.first; });
		for (size_t i = 0; i < std::min<size_t>(top.size(), 8); i++) {
			const auto& [count, key] = top[i];
			::printf("  %5" PRIu64 "x 0x%010" PRIx64 " %8" PRIu64 " KiB %4ux%-4u %s\n", count,
			         key.address, key.size / 1024, key.width, key.height,
			         vk::to_string(key.format).c_str());
		}
		std::fflush(stdout);
		m_created = m_created_bytes = m_deleted = m_deleted_bytes = 0;
		m_by_range.clear();
	}

private:
	struct Key {
		uint64_t   address;
		uint64_t   size;
		vk::Format format;
		uint32_t   width;
		uint32_t   height;
		auto       operator<=>(const Key&) const = default;
	};
	std::mutex                            m_mutex;
	std::map<Key, uint64_t>               m_by_range;
	uint64_t                              m_created       = 0;
	uint64_t                              m_created_bytes = 0;
	uint64_t                              m_deleted       = 0;
	uint64_t                              m_deleted_bytes = 0;
	std::chrono::steady_clock::time_point m_report_time   = std::chrono::steady_clock::now();
};

ImageChurnStats g_image_churn_stats;

} // namespace

ImageId TextureCache::InsertImage(const ImageInfo& info, std::source_location where) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	g_image_churn_stats.Note(info, true);
	m_slot_images[id].created_frame = DrawDiff::Frame();
	m_slot_images[id].created_line  = where.line();
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	return id;
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
	image.registered = true;
	NoteBindlessStateChange(image);
	image.lru_id     = m_lru_cache.Insert(id, LruClock());
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (image.bindless_pinned) {
		NoteBindlessStateChange(image);
		image.bindless_pinned = false;
		if (on_bindless_unregister) {
			on_bindless_unregister(id);
		}
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
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	// Only depth images own stencil associations (AssociateStencil); the scan visits every image.
	if (!image->depth_id && image->info.IsDepth()) {
		std::vector<ImageId> associations;
		m_slot_images.ForEach([&](ImageId candidate, const Image& associated) {
			if (associated.depth_id == id) {
				associations.push_back(candidate);
			}
		});
		for (const auto association: associations) {
			FreeImage(association);
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	g_image_churn_stats.Note(image->info, false);
	if (m_graphics.address_binding_report_enabled && image->backing.image != nullptr) {
		AddressBindingAnnotate(reinterpret_cast<uint64_t>(static_cast<VkImage>(image->backing.image)),
		                       image->bindless_pinned ? "deleted while bindless-pinned"
		                                              : "deleted");
	}
	m_download_images.erase(id);
	if (image->info.HasMetadata()) {
		const auto metadata = m_surface_metas.find(image->info.metadata.range.address);
		if (metadata != m_surface_metas.end() &&
		    image->info.metadata.kind == ImageMetadataKind::Htile &&
		    metadata->second.type == MetaDataInfo::Type::HTile) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
		}
	}
	UnregisterImage(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_images.erase(id); });
	} else {
		m_slot_images.erase(id);
	}
}

// KYTY_IMAGE_STATS also prints the source lines that free images ("Image frees (5 s)"), with how
// many still held GPU-written contents the free discarded.
static void NoteImageFree(const std::source_location& where, bool gpu_modified) {
	static auto& stats = Common::LiveSwitches::Get("KYTY_IMAGE_STATS", 0);
	if (stats.load(std::memory_order_relaxed) == 0) {
		return;
	}
	static std::mutex                                               mutex;
	static std::map<uint32_t, std::pair<uint64_t, uint64_t>>        sites;
	static auto report = std::chrono::steady_clock::now();
	std::lock_guard lock(mutex);
	auto& e = sites[where.line()];
	e.first++;
	e.second += gpu_modified ? 1 : 0;
	const auto now = std::chrono::steady_clock::now();
	if (now - report >= std::chrono::seconds(5)) {
		report = now;
		::printf("Image frees (5 s) by textureCache.cpp line:");
		for (const auto& [line, v]: sites) {
			::printf(" %u:%llu(gpu %llu)", line, static_cast<unsigned long long>(v.first),
			         static_cast<unsigned long long>(v.second));
		}
		::printf("\n");
		std::fflush(stdout);
		sites.clear();
	}
}

// Research: KYTY_FREE_WRITEBACK (live, default 0). Freeing an image the GPU wrote dropped its
// contents: guest memory kept the bytes from before, and the next image created over the range
// (another format/size the game aliases there) read those. On the console the memory holds what
// the GPU wrote. 1 writes the contents back to guest memory first (asynchronous download);
// 2 also waits for that download, so an image created next uploads the final bytes.
void TextureCache::FreeImage(ImageId id, std::source_location where) {
	auto& image = m_slot_images[id];
	const bool gpu_modified = image.IsGpuModified();
	NoteImageFree(where, gpu_modified);
	if (gpu_modified) {
		DrawDiff::NoteGpuFree(image.info.data.address);
	}
	if (gpu_modified) {
		DumpImage(id, "free");
	}
	if (gpu_modified) {
		static auto& writeback = Common::LiveSwitches::Get("KYTY_FREE_WRITEBACK", 0);
		const auto   mode      = writeback.load(std::memory_order_relaxed);
		// 3: copy the contents into the buffer cache's GPU copy of the range, on the GPU and in
		// stream order, and mark those bytes GPU-written: an image created over the range next
		// (FindImage: HasGpuDirtyBytes -> MarkBufferModified) uploads them from that buffer, as
		// the console's next pass would read the memory. No CPU round trip, no wait.
		if (mode == 3) {
			(void)WritebackToBufferCache(id);
		} else if (mode != 0 && mode != 3 && m_scheduler.Active() && DownloadImageMemory(id)) {
			if (mode >= 2) {
				const auto tick = m_scheduler.CurrentTick();
				m_scheduler.Flush();
				m_scheduler.Wait(tick);
				m_scheduler.WaitPriorityOperations(tick);
			}
		}
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, LruClock());
	}
}

void TextureCache::TouchImages(std::span<const ImageId> ids) {
	std::scoped_lock lock {m_lock};
	for (const auto id: ids) {
		if (auto* image = m_slot_images.try_get(id); image != nullptr) {
			TouchImage(*image);
		}
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
	const auto key = std::pair {desc.info.pixel_format, desc.info.type};
	if (const auto found = m_null_images.find(key); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = desc.info.type;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(key, id);
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
	std::vector<vk::BufferImageCopy> depth_copies(layers);
	std::vector<vk::BufferImageCopy> color_copies(layers);
	for (uint32_t layer = 0; layer < layers; layer++) {
		depth_copies[layer].bufferOffset      = depth_slice * layer;
		depth_copies[layer].bufferRowLength   = depth.info.pitch;
		depth_copies[layer].bufferImageHeight = depth.info.extent.height;
		depth_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, layer, 1};
		depth_copies[layer].imageExtent       = depth.info.extent;
		color_copies[layer].bufferOffset      = color_slice * layer;
		color_copies[layer].bufferRowLength   = color.info.pitch;
		color_copies[layer].bufferImageHeight = color.info.extent.height;
		color_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, layer, 1};
		color_copies[layer].imageExtent       = color.info.extent;
	}

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
		destination.CopyImageWithBuffer(source, copy_buffer);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
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
	// PPSA04264
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
	    !cached.info.HasMetadata();
	bool recreate = cached.info.resources < requested.resources;
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
		info.mip_layout = cached.info.mip_layout;
	} else {
		info.resources = std::max(requested.resources, cached.info.resources);
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
		     requested.mip_layout != cached.info.mip_layout)) {
			// Research: KYTY_ALIAS_TILE_COPY (live, default 0). Wolverine renders its quarter-
			// resolution buffers (416x234 R32Sfloat / R16G16Sfloat at one address) with two tile
			// modes alternately, every frame. Freeing the cached image dropped what the GPU had
			// just drawn and the new one started from stale guest memory (ground and shadow
			// flicker suspect). With the switch on, an image of the same extent, levels, block
			// size and a view-compatible format is replaced by a copy of its texels instead.
			static auto& alias_copy = Common::LiveSwitches::Get("KYTY_ALIAS_TILE_COPY", 0);
			if (alias_copy.load(std::memory_order_relaxed) != 0 && !merged_id &&
			    requested.tile_mode != cached.info.tile_mode && cached.IsGpuModified() &&
			    !requested.IsDepth() && !cached.info.IsDepth() && requested.samples == 1 &&
			    cached.info.samples == 1 && cached.backing.samples == 1 &&
			    requested.extent == cached.info.extent &&
			    requested.resources == cached.info.resources && requested.type == cached.info.type &&
			    !requested.HasMetadata() && !cached.info.HasMetadata()) {
				RefreshImage(cached_id);
				auto       info           = requested;
				info.htile_clear_mask     = 0;
				const auto replacement_id = InsertImage(info);
				auto&      replacement    = m_slot_images[replacement_id];
				auto&      source         = m_slot_images[cached_id];
				if (ImageViewOps::FormatsCompatible(source.backing.format, replacement.backing.format)) {
					replacement.usage = source.usage;
					if (source.binding.is_bound || source.binding.is_target) {
						source.binding.needs_rebind = true;
					}
					CopyImage(replacement_id, cached_id);
					FreeImage(cached_id);
					return {replacement_id};
				}
				FreeImage(replacement_id);
			}
			if (safe_to_delete) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.data.size == cached.info.data.size &&
		    (requested.IsVolume() || cached.info.IsVolume())) {
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
		if (cached.binding.is_target && !m_slot_images[merged_id].binding.is_target) {
			NoteBindlessStateChange(m_slot_images[merged_id]);
		}
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
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

struct TextureCache::TextureTransfer {
	TextureUploadLayout              layout;
	std::vector<vk::BufferImageCopy> regions;
	std::vector<GpuTileInfo>         tiles;
	bool                             swap_bgra16 = false;
	bool                             valid       = false;

	[[nodiscard]] uint64_t LinearSize() const {
		uint64_t size = 0;
		for (const auto& tile: tiles) {
			size = std::max(size, tile.linear_offset + tile.linear_size);
		}
		return size;
	}
};

struct TextureCache::ImageDownload {
	TextureTransfer texture;
	bool                depth_target = false;
	bool                valid        = false;
};

TextureCache::TextureTransfer
TextureCache::BuildTextureTransfer(const Image& image, BindingType binding,
                                    TransferDirection direction) const {
	const auto& info             = image.info;
	const bool  upload           = direction == TransferDirection::Upload;
	const bool  render_target    = binding == BindingType::RenderTarget;
	const bool  video_out        = binding == BindingType::VideoOut;
	uint32_t    layers           = info.TransferLayers();
	bool        volume           = info.IsVolume();
	const char* owner            = "TextureCache readback";

	TextureTransfer transfer;
	transfer.swap_bgra16 = info.bgra16;
	if (upload) {
		if ((render_target || video_out) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		owner = "TextureCache";
		if (render_target) {
			owner = "RenderTarget";
		} else if (binding == BindingType::Storage) {
			owner = "StorageTextureCache";
		} else if (video_out) {
			if (info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("TextureCache: invalid color-attachment upload\n");
			}
			layers = info.resources.layers;
			volume = false;
			owner  = "VideoOut";
		}
	}

	transfer.layout = TextureCalcUploadLayout(info.guest_format, info.extent.width,
	                                         info.extent.height, info.resources.levels, layers,
	                                         info.tile_mode, info.data.size, volume, owner);
	transfer.regions = TextureBuildImageCopies(transfer.layout);
	if (info.IsDepth()) {
		for (auto& region: transfer.regions) {
			region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
		}
	}
	if (transfer.layout.surface.description.tile_mode != Prospero::TileMode::kLinear) {
		if (!TextureBuildGpuTileInfos(info.data.size, transfer.regions, transfer.layout,
		                              info.resources.levels, transfer.tiles)) {
			return transfer;
		}
	}
	transfer.valid = true;
	return transfer;
}

TextureCache::ImageDownload TextureCache::BuildDownload(const Image& image) const {
	const auto&  info    = image.info;
	const auto   binding = UploadBinding(image);
	ImageDownload transfer {.depth_target = binding == BindingType::DepthTarget};
	if (info.samples != 1 || image.backing.samples != 1) {
		return transfer;
	}
	if (transfer.depth_target) {
		transfer.valid = IsSupportedDepthPlaneReadback(info) && info.resources.layers != 0 &&
		             info.data.size % info.resources.layers == 0 &&
		             Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
		return transfer;
	}
	if (info.metadata.compression != VideoOutCompression::Uncompressed) {
		return transfer;
	}
	transfer.texture = BuildTextureTransfer(image, binding, TransferDirection::Download);
	transfer.valid   = transfer.texture.valid;
	return transfer;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto binding = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	const auto  upload  = [&](std::vector<vk::BufferImageCopy>& copies, TileManager::Result linear) {
		for (auto& copy: copies) {
			copy.bufferOffset += linear.offset;
		}
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	};

	if (binding != BindingType::DepthTarget) {
		const auto& info = image.info;
		auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		if (!transfer.valid) {
			EXIT("TextureCache: invalid texture upload: binding=%u addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 " format=%u tile=%u family=%u extent=%ux%ux%u "
			     "pitch=%u levels=%u layers=%u samples=%u\n",
			     static_cast<uint32_t>(binding), info.data.address, info.data.size,
			     static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
			     static_cast<uint32_t>(transfer.layout.surface.texture.block.family), info.extent.width,
			     info.extent.height, info.extent.depth, info.pitch, info.resources.levels,
			     info.resources.layers, info.samples);
		}
		TileManager::Result linear {source.Handle(), source_offset, info.data.size};
		if (!transfer.tiles.empty()) {
			linear = m_tiler.Detile(source.Handle(), source_offset, info.data.size,
			                        transfer.LinearSize(), transfer.tiles);
		}
		if (transfer.swap_bgra16) {
			linear = m_tiler.SwapBgra16(linear);
		}
		upload(transfer.regions, linear);
		return;
	}

	// The stencil plane has its own row pitch and shares the native image
	// with the depth plane.
	auto info = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.bytes_per_block = 1;
		if (info.IsTiled()) info.pitch = TileGetDepthPitch(info.extent.width, 1, 0);
	}
	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto          layers          = info.resources.layers;
	const auto          full_slice_size = info.data.size / layers;
	auto copies = BuildDepthCopies(info, full_slice_size, image.depth_id
	                                                        ? vk::ImageAspectFlagBits::eStencil
	                                                        : vk::ImageAspectFlagBits::eDepth);
	TileManager::Result linear {source.Handle(), source_offset, source.Size() - source_offset};
	if (info.IsTiled()) {
		const auto tiles = BuildDepthTiles(info);
		linear =
		    m_tiler.Detile(source.Handle(), source_offset, info.data.size, info.data.size, tiles);
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
		for (uint32_t layer = 0; layer < layers; layer++) {
			copies[layer].bufferOffset = transfer_slice * layer;
		}
	}
	upload(copies, linear);
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

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range = desc.info.metadata.range;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile/CMask/FMask clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return;
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
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	if (m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		m_buffer_cache.ReadMemory(range.address, range.size, false);
	}
	const auto slice_size = range.size / layers;
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
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
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, slice_size);
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
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
	record.depth_id = depth_id;
	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		const auto       candidates =
		    FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			if (SameBacking(image.info, desc.info, exact_format)) {
				result = id;
			}
		}

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
		}

		if (result) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result);
				result = {};
			}
		}
		if (!result) {
			// KYTY_FREE_WRITEBACK=3: an image that stays alive but overlaps the new one with
			// another layout (Wolverine: RGBA8 832x468 and R8 416x234 at one address) also hands
			// its GPU-written bytes to the buffer cache, so the new image uploads them.
			if (DrawDiff::Enabled()) {
				std::string note;
				for (const auto candidate: candidates) {
					if (const auto* owner = m_slot_images.try_get(candidate); owner != nullptr) {
						char buf[160];
						std::snprintf(buf, sizeof(buf), "%s0x%010llx %ux%u %s tile=%u gpu=%d",
						              note.empty() ? "" : "; ",
						              static_cast<unsigned long long>(owner->info.data.address),
						              owner->info.extent.width, owner->info.extent.height,
						              vk::to_string(owner->info.pixel_format).c_str(),
						              static_cast<unsigned>(owner->info.tile_mode),
						              owner->IsGpuModified() ? 1 : 0);
						note += buf;
					}
				}
				char buf[96];
				std::snprintf(buf, sizeof(buf), " | new tile=%u", static_cast<unsigned>(desc.info.tile_mode));
				note += buf;
				DrawDiff::NoteInsert(desc.info.data.address, note.empty() ? "none" : note);
			}
			static auto& writeback = Common::LiveSwitches::Get("KYTY_FREE_WRITEBACK", 0);
			if (writeback.load(std::memory_order_relaxed) == 3) {
				for (const auto candidate: candidates) {
					if (const auto* owner = m_slot_images.try_get(candidate);
					    owner != nullptr && owner->IsGpuModified()) {
						(void)WritebackToBufferCache(candidate);
					}
				}
			}
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			} else if (writeback.load(std::memory_order_relaxed) == 3) {
				// The buffer cache got GPU contents for this range in the last 2 frames and nothing
				// the GPU wrote since is newer (the overlapping images are not GPU-modified): those
				// bytes, not guest memory, are current. Wolverine re-creates its R8 416x234 image
				// over an RGBA8 832x468 one up to 3 times a frame; 200 of 309 read stale memory.
				// Only an image at the very address written back, of a real color format: forcing
				// an upload for any overlapping image hit formats the upload path refuses (build
				// 4f7b4bf: fmt=0 1024x1024 linear, fatal).
				const auto it = m_written_back.find(inserted.info.data.address);
				if (it != m_written_back.end() && DrawDiff::Frame() - it->second <= 2 &&
				    inserted.info.pixel_format != vk::Format::eUndefined && !inserted.info.IsDepth() &&
				    inserted.info.IsTiled()) {
					inserted.MarkBufferModified();
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
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
		if (desc.type == BindingType::Texture && DrawDiff::Enabled() &&
		    image.created_frame == DrawDiff::Frame() && !image.IsGpuModified()) {
			DrawDiff::NoteFreshRead(image.info.data.address,
			                        vk::to_string(image.info.pixel_format).c_str(),
			                        image.info.extent.width, image.info.extent.height,
			                        image.IsBufferModified(), image.created_line);
		}
	}
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
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
	// Only an image that starts at the address qualifies, and it covers that byte: one page to
	// query instead of every page of the range (a multi-megabyte texel buffer on every draw).
	for (const auto id: FindImagesInRegion(address, 1, false)) {
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
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
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
	if (desc.type == BindingType::Texture) {
		DumpImage(id, "read");
		if (DrawDiff::Enabled() && image.last_write_frame != ~0ull &&
		    DrawDiff::Frame() > image.last_write_frame && DrawDiff::Frame() - image.last_write_frame <= 2) {
			DrawDiff::NoteHistoryRead(DrawDiff::t_shader, image.info.data.address,
			                          vk::to_string(image.info.pixel_format).c_str(),
			                          image.info.extent.width, image.info.extent.height,
			                          image.last_writer, DrawDiff::Frame() - image.last_write_frame);
		}
	} else if (desc.type == BindingType::Storage) {
		DumpImage(id, "storage-bind");
		image.last_write_frame = DrawDiff::Frame();
		image.last_writer      = DrawDiff::t_shader;
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
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.render_target = true;
	DumpImage(id, "target-before");
	image.last_write_frame    = DrawDiff::Frame();
	image.last_writer         = DrawDiff::t_shader;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.depth_target = true;
	image.last_write_frame   = DrawDiff::Frame();
	image.last_writer        = DrawDiff::t_shader;
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		m_surface_metas.emplace(desc.info.metadata.range.address,
		                        MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                                      .clear_mask = image.info.htile_clear_mask});
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		RefreshImage(AssociateStencil(id, desc.info.stencil));
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
	image.MarkGpuModified();
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock     lock {m_lock};
	ImageId              selected {};
	vk::ImageAspectFlags aspect {};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate    = vk::ImageAspectFlagBits::eStencil;
			candidate_id = owner->depth_id;
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
		if (selected && selected != candidate_id) {
			return false;
		}
		selected = candidate_id;
		aspect   = candidate;
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
	           {aspect, 0, image.info.resources.levels, 0, image.info.TransferLayers()}, clear);
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
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, command.Recorder());
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
		command.Recorder().beginRendering(&rendering);
		command.Recorder().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Recorder());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Recorder().clearColorImage(image.backing.image,
		                                   vk::ImageLayout::eTransferDstOptimal, &clear.color, 1,
		                                   &native_range);
	} else {
		command.Recorder().clearDepthStencilImage(image.backing.image,
		                                          vk::ImageLayout::eTransferDstOptimal,
		                                          &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.data.size / layers;
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
	auto copies = BuildDepthCopies(info, full_slice_size, vk::ImageAspectFlagBits::eDepth);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			for (auto& copy: copies) {
				copy.bufferOffset += destination_offset;
			}
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
			return;
		}
		const auto tiles = BuildDepthTiles(info);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, info.data.size,
		                  info.data.size, tiles);
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	for (uint32_t layer = 0; layer < layers; layer++) {
		copies[layer].bufferOffset = transfer_slice * layer;
	}
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto       guest_linear = tiled ? m_tiler.GetScratchBuffer(info.data.size, host_linear.buffer)
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
	if (!tiled) {
		return;
	}
	const auto tiles = BuildDepthTiles(info);
	m_tiler.Tile(guest_linear.buffer, guest_linear.offset, info.data.size, destination.Handle(),
	             destination_offset, info.data.size, tiles);
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                     uint64_t destination_size, ImageDownload transfer) {
	if (!transfer.valid) {
		EXIT("TextureCache: invalid image download transfer\n");
	}
	if (transfer.depth_target) {
		if (destination_size != image.info.data.size) {
			EXIT("TextureCache: partial depth image download is unsupported\n");
		}
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	auto&      texture   = transfer.texture;
	const auto transform = texture.swap_bgra16 ? TileManager::ColorTransform::SwapBgra16
	                                           : TileManager::ColorTransform::None;
	if (texture.tiles.empty()) {
		if (transform == TileManager::ColorTransform::SwapBgra16) {
			auto linear = m_tiler.GetScratchBuffer(destination_size);
			image.Download(texture.regions, linear.buffer, 0, linear.size);
			m_tiler.SwapBgra16(linear,
			                   {destination.Handle(), destination_offset, destination_size});
			return;
		}
		for (auto& copy: texture.regions) {
			copy.bufferOffset += destination_offset;
		}
		image.Download(texture.regions, destination.Handle(), destination_offset, destination_size);
		return;
	}

	m_tiler.TileImage(image, texture.regions, destination.Handle(), destination_offset,
	                  destination_size, texture.LinearSize(), texture.tiles, transform);
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
	if (image.info.IsVolume()) {
		// Volume mips contain strided block slices, so a mip's linear span cannot prove that
		// every retained slice fits. Keep volume synchronization whole-image only.
		if (!buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
			return false;
		}
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
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
	auto transfer = m_texture_cache.BuildDownload(image);
	if (!transfer.valid) {
		return false;
	}
	if (transfer.depth_target && copy_size != image.info.data.size) {
		return false;
	}
	if (!transfer.depth_target && levels < image.info.resources.levels) {
		auto& texture = transfer.texture;
		std::erase_if(texture.regions, [levels](const vk::BufferImageCopy& region) {
			return region.imageSubresource.mipLevel >= levels;
		});
		if (texture.regions.empty()) {
			return false;
		}
		if (!texture.tiles.empty()) {
			texture.tiles.clear();
			if (!TextureBuildGpuTileInfos(copy_size, texture.regions, texture.layout, levels,
			                              texture.tiles)) {
				return false;
			}
		}
	}
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, std::move(transfer));
	buffer.last_gpu_write_tick = m_scheduler.CurrentTick();
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	auto transfer = BuildDownload(image);
	if (!transfer.valid || !image.SafeToDownload()) {
		return false;
	}
	const auto range = image.info.data;
	auto&      ring  = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	std::shared_ptr<Buffer> dedicated;
	Buffer*                 download = &ring;
	uint8_t*                mapped   = nullptr;
	uint64_t                offset   = 0;
	if (range.size > ring.Size()) {
		dedicated = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     AllFlags, range.size);
		download  = dedicated.get();
		mapped    = dedicated->Mapped().data();
		if (mapped == nullptr) {
			return false;
		}
	} else {
		std::tie(mapped, offset) =
		    ring.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
		if (mapped == nullptr) {
			EXIT("TextureCache: failed to map reusable download buffer\n");
		}
		ring.Commit();
	}
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	download->Flush(offset, range.size);

	DownloadImage(image, *download, offset, range.size, std::move(transfer));
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
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
	                                                 nullptr, 1, &barrier, 0, nullptr);
	m_scheduler.DeferPriorityOperation([download, dedicated, range, mapped, offset] {
		download->Invalidate(offset, range.size);
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
	});
	return true;
}

// Diagnostic: KYTY_DUMP_IMAGE=<guest address> (live, default 0 = off). Saves the contents of the
// image at exactly that address, in guest layout, to _TempData/dump/ at three points: when it is
// freed holding GPU-written data ("free"), when a shader binds it to sample ("read", after the
// upload that bind triggers, so what the shader will see) and when it is bound for writing
// ("storage-bind", the contents before that write). KYTY_DUMP_IMAGE_COUNT (live, default 24) caps
// the files per address. One "Image dump:" log line per file (frame, order, shader, state).
// Comparing a "free" with the next "read" shows whether a frame gets what the last one wrote.
void TextureCache::DumpImage(ImageId id, const char* kind) {
	static auto& address_switch = Common::LiveSwitches::Get("KYTY_DUMP_IMAGE", 0);
	// KYTY_DUMP_SHADER=<16-hex hash> (startup): every image that shader binds (sampled, storage,
	// color target) is dumped instead of one address. Not a live switch: hashes exceed int64.
	static const uint64_t shader_wanted = [] {
		const char* v = std::getenv("KYTY_DUMP_SHADER");
		return v != nullptr ? std::strtoull(v, nullptr, 16) : 0ull;
	}();
	auto&      image   = m_slot_images[id];
	const bool by_shader = shader_wanted != 0 && DrawDiff::t_shader == shader_wanted;
	const auto wanted  = by_shader ? image.info.data.address
	                               : static_cast<uint64_t>(address_switch.load(std::memory_order_relaxed));
	if (wanted == 0) {
		return;
	}
	if (image.info.data.address != wanted || image.info.data.Empty() || image.depth_id) {
		return;
	}
	static auto&    count_switch = Common::LiveSwitches::Get("KYTY_DUMP_IMAGE_COUNT", 24);
	static uint64_t last_address = 0;
	static int64_t  written      = 0;
	static uint64_t sequence     = 0;
	if (!by_shader && last_address != wanted) {
		last_address = wanted;
		written      = 0;
	}
	if (written >= count_switch.load(std::memory_order_relaxed)) {
		return;
	}
	// KYTY_DUMP_IMAGE_FIRST=1 (live): only the first bind of the image in each frame, i.e. what the frame
	// starts from; consecutive files then show whether a frame continues from what the previous one left.
	static auto&    first_switch = Common::LiveSwitches::Get("KYTY_DUMP_IMAGE_FIRST", 0);
	static uint64_t last_frame   = UINT64_MAX;
	if (!by_shader && first_switch.load(std::memory_order_relaxed) != 0) {
		if (DrawDiff::Frame() == last_frame) return;
		last_frame = DrawDiff::Frame();
	}
	const auto frame  = DrawDiff::Frame();
	const auto shader = DrawDiff::t_shader;
	const auto order  = sequence++;
	char       name[256];
	std::snprintf(name, sizeof(name), "_TempData/dump/f%06llu_%04llu_%s_%010llx_%s_%ux%u_t%u.bin",
	              static_cast<unsigned long long>(frame), static_cast<unsigned long long>(order),
	              kind, static_cast<unsigned long long>(wanted),
	              vk::to_string(image.info.pixel_format).c_str(), image.info.extent.width,
	              image.info.extent.height, static_cast<unsigned>(image.info.tile_mode));
	// A sampled image the GPU never wrote (an alias re-created over another one) holds what its
	// upload put there once RefreshImage is done: dump that too, it is what the shader reads.
	const bool  uploaded = !image.IsBufferModified() && !image.IsCpuDirty();
	const bool  readable = std::strcmp(kind, "read") == 0 ? uploaded : image.SafeToDownload();
	const char* skipped  = !m_scheduler.Active() ? "no active command buffer"
	                       : !readable           ? "not safe to download"
	                                             : nullptr;
	ImageDownload transfer {};
	if (skipped == nullptr) {
		transfer = BuildDownload(image);
		if (!transfer.valid) {
			skipped = "no download transfer";
		}
	}
	::printf("Image dump: %s frame=%llu order=%llu shader=%016llx created_frame=%llu gpu=%d buffer=%d%s%s\n",
	         name, static_cast<unsigned long long>(frame), static_cast<unsigned long long>(order),
	         static_cast<unsigned long long>(shader),
	         static_cast<unsigned long long>(image.created_frame), image.IsGpuModified() ? 1 : 0,
	         image.IsBufferModified() ? 1 : 0, skipped != nullptr ? " SKIPPED: " : "",
	         skipped != nullptr ? skipped : "");
	std::fflush(stdout);
	if (skipped != nullptr) {
		return;
	}
	written++;
	const auto size     = image.info.data.size;
	auto       download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download,
	                                               0, AllFlags, size);
	auto*      mapped   = download->Mapped().data();
	if (mapped == nullptr) {
		return;
	}
	std::memset(mapped, 0, size);
	download->Flush(0, size);
	DownloadImage(image, *download, 0, size, std::move(transfer));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = 0;
	barrier.size                = size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
	                                                 nullptr, 1, &barrier, 0, nullptr);
	std::string path = name;
	m_scheduler.DeferPriorityOperation([download, mapped, size, path] {
		download->Invalidate(0, size);
		std::error_code error;
		std::filesystem::create_directories("_TempData/dump", error);
		if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
			std::fwrite(mapped, 1, size, file);
			std::fclose(file);
		}
	});
}

// KYTY_FREE_WRITEBACK=3 (see FreeImage).
bool TextureCache::WritebackToBufferCache(ImageId id) {
	auto& image = m_slot_images[id];
	if (!m_scheduler.Active() || !image.SafeToDownload() || image.info.data.Empty()) {
		DrawDiff::NoteWriteback(image.info.data.address,
		                        !m_scheduler.Active() ? "skipped: no active command buffer"
		                        : !image.SafeToDownload() ? "skipped: not safe to download"
		                                                  : "skipped: empty range");
		return false;
	}
	// Only plain committed guest memory: marking other ranges GPU-written made a later readback
	// write into memory without direct backing (build 7557eb1, fatal during loading).
	// Readbacks download aligned 2 MiB windows around a fault, so the whole window must be backed.
	constexpr uint64_t Window = 2ull * 1024 * 1024;
	const uint64_t     window_begin = image.info.data.address & ~(Window - 1);
	const uint64_t     window_end =
	    (image.info.data.address + image.info.data.size + Window - 1) & ~(Window - 1);
	if (!LibKernel::Memory::IsCommittedRange(window_begin, window_end - window_begin)) {
		DrawDiff::NoteWriteback(image.info.data.address, "skipped: 2 MiB window not committed");
		return false;
	}
	auto transfer = BuildDownload(image);
	if (!transfer.valid || transfer.depth_target) {
		DrawDiff::NoteWriteback(image.info.data.address,
		                        transfer.depth_target ? "skipped: depth" : "skipped: no download transfer");
		return false;
	}
	const auto [buffer, offset] = m_buffer_cache.ObtainBuffer(
	    image.info.data.address, image.info.data.size, true, false, BufferId {}, false);
	if (buffer == nullptr) {
		DrawDiff::NoteWriteback(image.info.data.address, "skipped: no buffer");
		return false;
	}
	DrawDiff::NoteWriteback(image.info.data.address, "written back");
	m_written_back[image.info.data.address] = DrawDiff::Frame();
	if (m_written_back.size() > 4096) {
		std::erase_if(m_written_back, [](const auto& e) { return DrawDiff::Frame() - e.second > 2; });
	}
	DownloadImage(image, *buffer, offset, image.info.data.size, std::move(transfer));
	buffer->last_gpu_write_tick = m_scheduler.CurrentTick();
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	m_scheduler.EndRendering();
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                                                 vk::PipelineStageFlagBits::eAllCommands, {}, 1,
	                                                 &after, 0, nullptr, 0, nullptr);
	return true;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
}

bool TextureCache::HasImagesInRegion(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

bool TextureCache::IsRegionGpuModified(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		// PPSA17168: S_LOAD_DWORD reads shader data at an address overlapping an old
		// render target whose memory the CPU has reused. The cached image still retains
		// its earlier GPU-modified flag.
		if (!image.depth_id && image.IsGpuModified() && !image.IsDefinitelyCpuDirty()) {
			return true;
		}
	}
	return false;
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

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	return (found->second.clear_mask & (1u << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	if (is_clear) {
		found->second.clear_mask |= 1u << slice;
	} else {
		found->second.clear_mask &= ~(1u << slice);
	}
	return true;
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
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

uint64_t TextureCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	m_gc_tick++;
	const uint64_t clock    = LruClock();
	static auto&   combined = Common::LiveSwitches::Get("KYTY_GC_COMBINED", 1);
	m_other_cache_memory =
	    combined.load(std::memory_order_relaxed) != 0 ? m_buffer_cache.UsedMemory() : 0;
	// KYTY_GC_HEADROOM_MB (live, default 0): count this much more as used, so both collectors
	// trim earlier. The thresholds come from the device budget at startup, but ~2.2 GiB outside
	// the caches (staging rings, pipelines, ...) put the process over its budget while the caches
	// were still below their critical mark (Wolverine run 31: device 9.89 of 9.88 GiB).
	static auto& headroom = Common::LiveSwitches::Get("KYTY_GC_HEADROOM_MB", 0);
	m_gc_headroom =
	    static_cast<uint64_t>(std::max<int64_t>(0, headroom.load(std::memory_order_relaxed)))
	    << 20u;
	ReportGcStats(clock);
	if (GcUsedMemory() < m_trigger_gc_memory) {
		return;
	}
	m_gc_stats.runs++;
	if (m_gc_budget_frame != clock) {
		m_gc_budget_frame       = clock;
		m_gc_freed_bytes_frame  = 0;
		m_gc_freed_images_frame = 0;
		m_gc_written_back_bytes_frame = 0;
	}
	// Within 512 MiB of what the driver lets the process keep in VRAM, unsampled bindless
	// textures go after KYTY_BINDLESS_EVICT_TIGHT_AGE frames: a camera turn streams textures in
	// faster than the normal age frees them, and past the budget the driver pages memory out
	// (Wolverine run 53: 2.5 GiB to system memory, second-long frames).
	const auto driver_budget      = m_graphics.GetDriverMemoryBudget();
	const bool near_driver_budget = driver_budget != 0 &&
	                                m_graphics.GetDeviceMemoryUsage() + (512ull << 20u) >= driver_budget;
	const auto collect = [&](bool allow_aggressive) {
		bool               pressured  = GcUsedMemory() >= m_pressure_gc_memory;
		bool               aggressive = allow_aggressive && GcUsedMemory() >= m_critical_gc_memory;
		// KYTY_IMAGE_GC_CRITICAL_AGE (live, frames, default 30): the age above the critical mark
		// (see KYTY_BUFFER_GC_CRITICAL_AGE). Wolverine run 51: once bindless images were aged,
		// age 1 reached render targets used every few frames and recreated 2.4 GiB per 5 s
		// (2-7 fps); age 30 recreated 0.3 GiB (14 fps).
		static auto&   critical_age   = Common::LiveSwitches::Get("KYTY_IMAGE_GC_CRITICAL_AGE", 30);
		const uint64_t aggressive_age = static_cast<uint64_t>(
		    std::max<int64_t>(1, critical_age.load(std::memory_order_relaxed)));
		const uint64_t age = std::min<uint64_t>(aggressive  ? aggressive_age
		                                        : pressured ? 4
		                                                    : 16,
		                                        clock);
		constexpr uint64_t MiB = 1024 * 1024;
		constexpr size_t   MaxFreesInFrame = 1024;
		uint64_t           byte_budget     = 0;
		size_t             deletions       = 10;
		if (pressured || aggressive) {
			const auto threshold = aggressive ? m_critical_gc_memory : m_pressure_gc_memory;
			const auto excess    = GcUsedMemory() - threshold;
			byte_budget = aggressive ? std::max<uint64_t>(64 * MiB, excess / 4)
			                         : std::max<uint64_t>(16 * MiB, excess / 8);
			if (m_gc_freed_bytes_frame >= byte_budget ||
			    m_gc_freed_images_frame >= MaxFreesInFrame) {
				return;
			}
			deletions = MaxFreesInFrame - m_gc_freed_images_frame;
		}
		const size_t         visit_limit = pressured || aggressive ? deletions * 4 : deletions;
		std::vector<ImageId> candidates;
		candidates.reserve(std::min<size_t>(visit_limit, 4096));
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(clock - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() >= visit_limit || candidates.size() >= 4096;
		});
		if (pressured || aggressive) {
			m_gc_stats.pressured++;
		}
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			if (!(pressured || aggressive)) {
				--deletions;
			}
			m_gc_stats.visited++;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				m_gc_stats.gone_or_depth++;
				continue;
			}
			// KYTY_BINDLESS_EVICT=1 (live, default 0): unpin bindless images the shaders have not
			// sampled for KYTY_BINDLESS_EVICT_AGE frames (live, default 60; TouchImages keeps their
			// age), or KYTY_BINDLESS_EVICT_TIGHT_AGE (live, default 15) near the driver budget. The
			// age stays well above the few frames feedback snapshots lag. Off by default: with it,
			// Wolverine showed wrong textures and lost the device (runs 54-58, a stale bindless
			// slot reference is suspected; see KYTY_BINDLESS_SLOT_REUSE).
			static auto& evict_pinned = Common::LiveSwitches::Get("KYTY_BINDLESS_EVICT", 0);
			static auto& evict_age    = Common::LiveSwitches::Get("KYTY_BINDLESS_EVICT_AGE", 60);
			static auto& tight_age = Common::LiveSwitches::Get("KYTY_BINDLESS_EVICT_TIGHT_AGE", 15);
			const auto   pinned_age = static_cast<uint64_t>(std::max<int64_t>(
			    8, (near_driver_budget ? tight_age : evict_age).load(std::memory_order_relaxed)));
			if (owner->bindless_pinned &&
			    (evict_pinned.load(std::memory_order_relaxed) == 0 ||
			     m_lru_cache.TickOf(owner->lru_id) + pinned_age > clock)) {
				m_gc_stats.pinned++;
				m_gc_stats.pinned_bytes += owner->AccountedSize();
				continue;
			}
			// Unpinning points its table slots back at the placeholder and makes its keys pending,
			// but keeps the image: it is collected like any other once unused for the normal age,
			// and a key sampled again in between resolves to it without a reload. Freeing it here
			// directly lost the device in Wolverine runs 54/55 once large images had dedicated
			// memory (a read of the freed image, ReadInvalid outside every buffer).
			if (owner->bindless_pinned) {
				if (m_graphics.address_binding_report_enabled) {
					AddressBindingAnnotate(
					    reinterpret_cast<uint64_t>(static_cast<VkImage>(owner->backing.image)),
					    "unpinned by the collector");
				}
				NoteBindlessStateChange(*owner);
				owner->bindless_pinned = false;
				if (on_bindless_unregister) {
					on_bindless_unregister(id);
				}
				TouchImage(*owner);
				m_gc_stats.unpinned++;
				m_gc_stats.unpinned_bytes += owner->AccountedSize();
				continue;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				constexpr uint64_t WriteBackPerFrame = 256 * MiB;
				if (safe && owner->info.IsTiled() &&
				    (!(pressured || aggressive) ||
				     (m_gc_written_back_bytes_frame != 0 &&
				      m_gc_written_back_bytes_frame + owner->info.data.size > WriteBackPerFrame))) {
					m_gc_stats.gpu_kept++;
					m_gc_stats.gpu_kept_bytes += owner->AccountedSize();
					continue;
				}
				if (safe && !pressured) {
					m_gc_stats.gpu_kept++;
					m_gc_stats.gpu_kept_bytes += owner->AccountedSize();
					continue;
				}
				if (safe && !DownloadImageMemory(id)) {
					m_gc_stats.download_failed++;
					continue;
				}
				if (safe && owner->info.IsTiled()) {
					m_gc_written_back_bytes_frame += owner->info.data.size;
				}
			}
			const auto freed = owner->AccountedSize();
			FreeImage(id);
			m_gc_stats.freed++;
			m_gc_stats.freed_bytes += freed;
			if (pressured || aggressive) {
				--deletions;
				m_gc_freed_images_frame++;
				m_gc_freed_bytes_frame += freed;
				if (m_gc_freed_bytes_frame >= byte_budget) {
					break;
				}
			}
			if (GcUsedMemory() < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (GcUsedMemory() < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (GcUsedMemory() >= m_critical_gc_memory) {
		collect(true);
	}
}

// KYTY_IMAGE_GC_STATS=1 (live): every 5 s, what the image collector visited, why it kept each
// candidate, and how many bytes bindless-pinned images hold (pinned images are never collected).
void TextureCache::ReportGcStats(uint64_t clock) {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_IMAGE_GC_STATS", 0);
	if (enabled.load(std::memory_order_relaxed) == 0) {
		m_gc_stats = {};
		return;
	}
	const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	if (m_gc_stats_report_us == 0) {
		m_gc_stats_report_us = static_cast<uint64_t>(now_us);
		return;
	}
	if (static_cast<uint64_t>(now_us) - m_gc_stats_report_us < 5'000'000) {
		return;
	}
	m_gc_stats_report_us = static_cast<uint64_t>(now_us);
	uint64_t images = 0;
	uint64_t pinned = 0;
	uint64_t pinned_bytes = 0;
	uint64_t pinned_old_bytes = 0;
	uint64_t old_bytes = 0;
	m_lru_cache.ForEachItemBelow(UINT64_MAX, [&](ImageId id) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr) {
			return;
		}
		images++;
		const auto size = image->AccountedSize();
		const bool old  = m_lru_cache.TickOf(image->lru_id) + 16 <= clock;
		old_bytes += old ? size : 0;
		if (image->bindless_pinned) {
			pinned++;
			pinned_bytes += size;
			pinned_old_bytes += old ? size : 0;
		}
	});
	constexpr double MiB = 1024.0 * 1024.0;
	const auto&      st  = m_gc_stats;
	::printf("ImageGC (5 s): runs %" PRIu64 " (pressured %" PRIu64 "), visited %" PRIu64
	         ", gone/depth %" PRIu64 ", pinned %" PRIu64 " (%.0f MiB), unpinned %" PRIu64
	         " (%.0f MiB), gpu kept %" PRIu64
	         " (%.0f MiB), download failed %" PRIu64 ", freed %" PRIu64 " (%.0f MiB)\n",
	         st.runs, st.pressured, st.visited, st.gone_or_depth, st.pinned,
	         static_cast<double>(st.pinned_bytes) / MiB, st.unpinned,
	         static_cast<double>(st.unpinned_bytes) / MiB, st.gpu_kept,
	         static_cast<double>(st.gpu_kept_bytes) / MiB, st.download_failed, st.freed,
	         static_cast<double>(st.freed_bytes) / MiB);
	::printf("ImageGC now: %" PRIu64 " images %.0f MiB (unused 16+ frames %.0f MiB), pinned %" PRIu64
	         " %.0f MiB (unused 16+ frames %.0f MiB)\n",
	         images, static_cast<double>(m_total_used_memory) / MiB,
	         static_cast<double>(old_bytes) / MiB, pinned, static_cast<double>(pinned_bytes) / MiB,
	         static_cast<double>(pinned_old_bytes) / MiB);
	std::fflush(stdout);
	m_gc_stats = {};
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
