#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "slow-log.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"
#include "live-counters.h"
#include "native-resource-state.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <tuple>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>
#include "speculation-state.h"

[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_write_window_handoff_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_image_granules_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_texture_resolve_pages_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_partial_image_dirty_mode {0};
// 1: a partial refresh uploads only the rows of tile blocks over the dirty ranges of a large
// subresource, not the whole subresource (UploadImagePartial).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_partial_row_bands_mode {0};

namespace Libs::Graphics {

namespace {

// Guest frames an overlapping image must stay unused before a lookup of another image may delete
// it. (Compared with scheduler ticks before, dozens a frame: a 4K target and a 2560x1440 one the
// game renders into the same transient memory every frame deleted and re-created each other,
// 64 MiB of page tracking both ways, ~54 times a second.)
constexpr uint64_t NumFramesBeforeRemoval = 32;

// KYTY_PARTIAL_IMAGE_DIRTY: a game streaming textures into a large array writes one layer at a
// time. The whole-image path stopped watching all of the image's pages on the first write and
// then re-watched, copied, detiled and uploaded all of it (a 352 MiB array: 20-70 ms on the
// render thread, several times a second while walking into a new area). Images at least this
// large keep watching everything but the written granules and upload only the subresources
// over them.
constexpr uint64_t PartialDirtyMinSize = uint64_t {16} << 20;
// A write fault releases the image's granule around it (granules start at the image start):
// a streamed layer faults a few times instead of once per 4 KiB page.
constexpr uint64_t PartialDirtyGranule = uint64_t {1} << 20;
// A refresh whose subresources cover more than this share of the image uploads all of it.
constexpr uint64_t PartialUploadMaxShare = 2;
// Staged runs of picked subresources closer than this are staged as one run.
constexpr uint64_t PartialStageGap = uint64_t {64} << 10;

[[nodiscard]] bool DecodeDccClear(const TextureCache::ImageDesc& desc, vk::Format format,
                                  uint32_t fill, vk::ClearColorValue& clear) {
	const auto code = static_cast<uint8_t>(fill);
	if (fill != static_cast<uint32_t>(code) * 0x01010101u) {
		return false;
	}
	const auto& metadata = desc.info.metadata;
	if (code == 0x20) {
		// Clear-to-register is a color-buffer operation; the texture pipe cannot decode it.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.dcc_clear_register_valid &&
		       DecodePackedColorClear(format, metadata.dcc_clear_word, clear);
	}
	if (code != 0x00 && code != 0x40 && code != 0x80 && code != 0xc0) {
		return false;
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

[[nodiscard]] const char* BindingTypeName(TextureCache::BindingType type) {
	switch (type) {
		case TextureCache::BindingType::Texture: return "Texture";
		case TextureCache::BindingType::Storage: return "StorageTexture";
		case TextureCache::BindingType::RenderTarget: return "ColorTarget";
		case TextureCache::BindingType::DepthTarget: return "DepthTarget";
		case TextureCache::BindingType::VideoOut: return "VideoOut";
	}
	return "Image";
}

void NameImageBinding(GraphicContext& graphics, Image& image, vk::ImageView view,
                      TextureCache::BindingType type, const ImageViewInfo& view_info) {
	const auto* role = BindingTypeName(type);
	SetVulkanObjectNameF(
	    graphics.device, image.backing.image,
	    "Kyty.{}.Image[guest=0x{:016x} size=0x{:x} extent={}x{}x{} format={} mips={} layers={} "
	    "samples={}]",
	    role, image.info.data.address, image.info.data.size, image.info.extent.width,
	    image.info.extent.height, image.info.extent.depth,
	    static_cast<uint32_t>(image.info.pixel_format), image.info.resources.levels,
	    image.info.resources.layers, image.info.samples);
	SetVulkanObjectNameF(
	    graphics.device, view,
	    "Kyty.{}.View[guest=0x{:016x} format={} aspect=0x{:x} mip={}+{} layer={}+{}]", role,
	    image.info.data.address, static_cast<uint32_t>(view_info.format),
	    static_cast<vk::ImageAspectFlags::MaskType>(view_info.aspect), view_info.base_level,
	    view_info.level_count, view_info.base_layer, view_info.layer_count);
}

[[nodiscard]] std::vector<vk::BufferImageCopy> BuildDepthCopies(const ImageInfo& info,
                                                                uint64_t         slice_stride) {
	std::vector<vk::BufferImageCopy> copies(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		auto& copy             = copies[layer];
		copy.bufferOffset      = slice_stride * layer;
		copy.bufferRowLength   = info.pitch;
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, layer, 1};
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
		UpdateGcThresholds();
	}
}

// The collection thresholds from the GPU's current memory budget. The budget Windows gives a process
// shrinks while other programs (a browser, an overlay, a recorder) take video memory: thresholds fixed
// at start-up then let the caches fill past it, and allocations spilled to system memory (16 GB GPUs
// reached 15 GB of a 14.8 GB budget). Re-read at each collection.
void TextureCache::UpdateGcThresholds() {
	constexpr int64_t GiB = 1024ll * 1024 * 1024;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min<int64_t>(budget, 8 * GiB);
	m_pressure_gc_memory = static_cast<uint64_t>(
	    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
	m_critical_gc_memory = static_cast<uint64_t>(
	    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
	m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	m_over_budget_memory = static_cast<uint64_t>(std::max<int64_t>(budget, 0));
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
	    (cached.type != requested.type && requested.extent != vk::Extent3D {1, 1, 1})) {
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

bool TextureCache::SafeToDownload(const Image& image) {
	if (!image.SafeToDownload()) {
		return false;
	}
	const auto range = image.info.data;
	return !m_buffer_cache.HasGpuDirtyBytes(range.address, range.size);
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	return id;
}

void TextureCache::StampRegistrationPages(const Image& image, uint64_t epoch) {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) return;
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) m_registration_pages[page] = epoch;
}

bool TextureCache::StillResolved(const Image& image, uint64_t serial, uint64_t epoch) const {
	return image.serial == serial && image.registered &&
	       !RegistrationsSince(image.info.data.address, image.info.data.size, epoch);
}

bool TextureCache::RegistrationsSince(uint64_t address, uint64_t size, uint64_t epoch) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) return true;
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		const auto* stamp = m_registration_pages.Find(page);
		if (stamp != nullptr && *stamp > epoch) return true;
	}
	return false;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	StampRegistrationPages(image, m_resolution_epoch.fetch_add(1, std::memory_order_release) + 1);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		m_image_page_table[page].push_back(id);
	}
	MarkImageGranules(image.info.data.address, image.info.data.size);
	if (m_image_starts[image.info.data.address]++ == 0) {
		const auto epoch = m_start_epoch.load(std::memory_order_relaxed) + 1;
		if (m_start_log.size() >= 8192) m_start_log.erase(m_start_log.begin(), m_start_log.begin() + 4096);
		m_start_log.emplace_back(epoch, image.info.data.address);
		m_start_epoch.store(epoch, std::memory_order_release);
	}
	image.registered = true;
	++image.registrations;
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	image.lru_tick   = m_gc_tick;
	m_total_used_memory += image.AccountedSize();
	m_cache_bytes += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	StampRegistrationPages(image, m_resolution_epoch.fetch_add(1, std::memory_order_release) + 1);
	++m_image_granule_releases;
	UntrackImage(id, "unregister");
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
	}
	if (const auto start = m_image_starts.find(image.info.data.address);
	    start == m_image_starts.end() || start->second == 0) {
		EXIT("TextureCache: image missing from start index\n");
	} else if (--start->second == 0) {
		m_image_starts.erase(start);
	}
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	m_cache_bytes -= std::min(m_cache_bytes, accounted);
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	m_partial_plans.erase(image->serial);
	if (!image->depth_id) {
		std::vector<ImageId> associations;
		for (const auto candidate: m_stencil_associations) {
			const auto* associated = m_slot_images.try_get(candidate);
			if (associated != nullptr && associated->depth_id == id) {
				associations.push_back(candidate);
			}
		}
		for (const auto association: associations) {
			auto& associated = m_slot_images[association];
			if (associated.IsGpuModified()) {
				associated.ClearGpuModified();
			}
			DeleteImage(association);
		}
	} else {
		std::erase(m_stencil_associations, id);
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image->info.HasMetadata()) {
		const auto metadata = m_surface_metas.find(image->info.metadata.range.address);
		if (metadata != m_surface_metas.end() &&
		    ((image->info.metadata.kind == ImageMetadataKind::Dcc &&
		      metadata->second.type == MetaDataInfo::Type::Dcc) ||
		     (image->info.metadata.kind == ImageMetadataKind::Htile &&
		      metadata->second.type == MetaDataInfo::Type::HTile))) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
			m_meta_epoch.fetch_add(1, std::memory_order_release);
		}
	}
	UnregisterImage(id);
	// (A speculation's packet may still read it: Spec::PacketNow.)
	const auto packet = Spec::PacketNow();
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id, packet] {
			m_retired.emplace_back(id, packet);
			EraseRetired();
		});
	} else {
		m_retired.emplace_back(id, packet);
		EraseRetired();
	}
}

[[gnu::noinline]] void TextureCache::FreeImage(ImageId id, const char* site) {
	auto& image = m_slot_images[id];
	if (SlowLog::Threshold() > 0.0 && image.info.data.size >= PartialDirtyMinSize) {
		std::printf("[tsc %llu] FREE %s addr=0x%llx size=0x%llx fmt=%u %ux%u layers=%u levels=%u tile=%u target=%d "
		            "gpu=%d\n",
		            static_cast<unsigned long long>(__rdtsc()), site,
		            static_cast<unsigned long long>(image.info.data.address),
		            static_cast<unsigned long long>(image.info.data.size),
		            static_cast<uint32_t>(image.info.guest_format), image.info.extent.width,
		            image.info.extent.height, image.info.resources.layers, image.info.resources.levels,
		            static_cast<uint32_t>(image.info.tile_mode), image.binding.is_target ? 1 : 0,
		            image.IsGpuModified() ? 1 : 0);
		std::fflush(stdout);
	}
	if (image.IsGpuModified()) {
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	if (auto* spec = Spec::Current()) { // (touched when the speculative translation is committed)
		auto& touched = spec->touched_images;
		if (touched.empty() || touched.back().first != &image) touched.emplace_back(&image, image.serial);
		return;
	}
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
		image.lru_tick = m_gc_tick;
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	if (image.NeedsMaybeCpuHash()) {
		image.SetMaybeCpuHash(image.HashGuestEdges());
	}
	UntrackImage(id, "maybe-dirty");
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	RetrackHoles(image);
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

void TextureCache::UntrackImage(ImageId id, const char* why) {
	auto& image = m_slot_images[id];
	if (SlowLog::Threshold() > 0.0 && image.IsTracked() && image.info.data.size >= PartialDirtyMinSize) {
		std::printf("[tsc %llu] UNTRACK %s addr=0x%llx size=0x%llx holes=%zu partial=%d\n",
		            static_cast<unsigned long long>(__rdtsc()), why,
		            static_cast<unsigned long long>(image.info.data.address),
		            static_cast<unsigned long long>(image.info.data.size), image.untracked_holes.size(),
		            image.IsPartiallyCpuDirty() ? 1 : 0);
	}
	// Unwatched pages no longer report writes: only a whole upload is complete again.
	image.DropPartialDirty();
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (!image.untracked_holes.empty()) {
		const auto holes = std::move(image.untracked_holes);
		image.untracked_holes.clear();
		auto       cursor = address & ~(TRACKER_PAGE_SIZE - 1);
		const auto end    = (address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
		for (const auto& [hole_begin, hole_end]: holes) {
			if (cursor < hole_begin) {
				m_page_manager.UpdatePageWatchers<false>(cursor, hole_begin - cursor);
			}
			cursor = std::max(cursor, hole_end);
		}
		if (cursor < end) {
			m_page_manager.UpdatePageWatchers<false>(cursor, end - cursor);
		}
		return;
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImagePages(Image& image, uint64_t begin, uint64_t end) {
	const auto tracked_begin = image.track_addr & ~(TRACKER_PAGE_SIZE - 1);
	const auto tracked_end =
	    (image.track_addr_end + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	begin = std::max(begin, tracked_begin);
	end   = std::min(end, tracked_end);
	if (begin >= end) {
		return;
	}
	auto cursor = begin;
	for (const auto& [hole_begin, hole_end]: image.untracked_holes) {
		if (hole_end <= cursor) {
			continue;
		}
		if (hole_begin >= end) {
			break;
		}
		if (cursor < hole_begin) {
			m_page_manager.UpdatePageWatchers<false>(cursor, hole_begin - cursor);
		}
		cursor = std::max(cursor, hole_end);
	}
	if (cursor < end) {
		m_page_manager.UpdatePageWatchers<false>(cursor, end - cursor);
	}
	Image::AddRange(image.untracked_holes, begin, end);
}

void TextureCache::RetrackHoles(Image& image) {
	if (image.untracked_holes.empty()) {
		return;
	}
	if (!image.IsTracked()) {
		EXIT("TextureCache: untracked image keeps page holes\n");
	}
	for (const auto& [hole_begin, hole_end]: image.untracked_holes) {
		m_page_manager.UpdatePageWatchers<true>(hole_begin, hole_end - hole_begin);
	}
	image.untracked_holes.clear();
}

void TextureCache::FinishRefresh(Image& image) {
	// Watch the released pages again before the image counts as clean.
	RetrackHoles(image);
	image.RefreshComplete();
}

bool TextureCache::PartialDirtyCandidate(const Image& image) {
	const auto& info = image.info;
	// Surface metadata (DCC of a color target) does not change how guest data uploads: the
	// whole-image path uploads the same subresources from the same guest bytes. A depth array
	// in a texture tile mode (the game's streamed shadow maps) uploads as a texture too; one
	// that uploads as a depth target takes the whole-image path at refresh (UploadImagePartial).
	return kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) != 0 &&
	       info.data.size >= PartialDirtyMinSize && !image.depth_id &&
	       image.backing.image != nullptr && info.samples == 1 && !info.IsVolume() &&
	       !info.HasStencil() && info.metadata.compression == VideoOutCompression::Uncompressed;
}

bool TextureCache::TryInvalidatePartial(Image& image, uint64_t address, uint64_t size,
                                        uint64_t granule) {
	const auto& info = image.info;
	if (!PartialDirtyCandidate(image) || !image.registered || !image.IsTracked() ||
	    image.track_addr != info.data.address || image.track_addr_end != info.data.End() ||
	    !image.CanTakePartialDirty() || image.IsGpuModified() || image.IsBufferModified() ||
	    image.IsStencilModified()) {
		return false;
	}
	const auto base  = info.data.address;
	const auto first = std::max(address, base) - base;
	const auto last  = std::min(address + size, info.data.End()) - base;
	if (first >= last) {
		return false;
	}
	const auto granule_begin = base + first / granule * granule;
	const auto granule_end   = base + std::min(info.data.size, ((last - 1) / granule + 1) * granule);
	const auto page_begin = granule_begin & ~(TRACKER_PAGE_SIZE - 1);
	const auto page_end   = (granule_end + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	// The released pages are dirty wherever they hold image bytes.
	image.InvalidateCpuWritePartial(std::max(page_begin, base), std::min(page_end, info.data.End()));
	UntrackImagePages(image, page_begin, page_end);
	LiveCounters::Add(LiveCounters::PartialDirtyFaults);
	return true;
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	if (!image.untracked_holes.empty()) {
		UntrackImage(id, "head");
		return;
	}
	const auto address = (begin + TRACKER_PAGE_SIZE) & ~(TRACKER_PAGE_SIZE - 1);
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
	if (!image.untracked_holes.empty()) {
		UntrackImage(id, "tail");
		return;
	}
	const auto address   = end & ~(TRACKER_PAGE_SIZE - 1);
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

namespace {
constexpr uint32_t ImageGranuleBits  = 16;
constexpr uint64_t ImageGranuleSpace = uint64_t {1} << 40; // the image page table's address space
} // namespace

void TextureCache::MarkImageGranules(uint64_t address, uint64_t size) const {
	if (size == 0 || address >= ImageGranuleSpace) return;
	if (m_image_granules.empty()) m_image_granules.assign((ImageGranuleSpace >> ImageGranuleBits) / 64, 0);
	const uint64_t last = (std::min(address + size, ImageGranuleSpace) - 1) >> ImageGranuleBits;
	for (uint64_t g = address >> ImageGranuleBits; g <= last; ++g) m_image_granules[g >> 6] |= uint64_t {1} << (g & 63);
}

bool TextureCache::MayHaveImages(uint64_t address, uint64_t size) const {
	if (kyty_local_image_granules_mode.load(std::memory_order_relaxed) == 0 || m_image_granules.empty() ||
	    size == 0 || address >= ImageGranuleSpace || size > ImageGranuleSpace - address)
		return true;
	if (m_image_granule_releases > 4096) {
		// Released images leave their bits; rebuild from the registered ones.
		std::fill(m_image_granules.begin(), m_image_granules.end(), 0);
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered) MarkImageGranules(image.info.data.address, image.info.data.size);
		});
		m_image_granule_releases = 0;
	}
	const uint64_t last = (address + size - 1) >> ImageGranuleBits;
	for (uint64_t g = address >> ImageGranuleBits; g <= last;) {
		const uint64_t bit  = g & 63;
		const uint64_t span = std::min<uint64_t>(64 - bit, last - g + 1);
		const uint64_t mask = (span == 64 ? ~uint64_t {0} : (uint64_t {1} << span) - 1) << bit;
		if ((m_image_granules[g >> 6] & mask) != 0) return true;
		g += span;
	}
	return false;
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}
	if (!MayHaveImages(address, size)) {
		return {};
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			continue;
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
	}
	return result;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	// The view's type needs an image of that type (NullTextureDesc: 1D, 2D or 3D).
	const auto type = desc.info.type == Prospero::ImageType::kColor1D || desc.info.type == Prospero::ImageType::kColor3D
	                      ? desc.info.type
	                      : Prospero::ImageType::kColor2D;
	const auto key  = std::pair {desc.info.pixel_format, type};
	if (const auto found = m_null_images.find(key); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = type;
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
		FinishRefresh(image);
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
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size);
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
	    source.backing.format == destination.backing.format ||
	    (!source_depth && !dest_depth &&
	     vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format));
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

static bool GrowImageToLayers(ImageInfo& info, uint32_t layers) {
	const auto current = info.resources.layers;
	if (info.resources.levels != 1 || current == 0 || layers <= current ||
	    info.mip_layout[0].offset != 0 || info.mip_layout[0].size != info.data.size) {
		return false;
	}
	const auto stretch = [&](GuestRange& range) {
		if (range.Empty()) {
			return true;
		}
		if (range.size % current != 0 || range.size / current > UINT64_MAX / layers) {
			return false;
		}
		range.size = range.size / current * layers;
		return true;
	};
	auto grown = info;
	if (!stretch(grown.data) || !stretch(grown.stencil) || !stretch(grown.metadata.range)) {
		return false;
	}
	grown.mip_layout[0].size = grown.data.size;
	grown.resources.layers   = layers;
	info                     = grown;
	return true;
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if (!cached.info.IsDepth() && !requested.IsDepth()) {
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
	    requested.tile_mode == cached.info.tile_mode && !requested.HasStencil() &&
	    !cached.info.HasStencil() && !requested.HasMetadata() && !cached.info.HasMetadata();
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
	    requested.tile_mode == cached.info.tile_mode && requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	bool recreate = !cached.info.resources.Contains(requested.resources);
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
	LiveCounters::Add(LiveCounters::DepthOverlaps);
	LiveCounters::Add(LiveCounters::DepthOverlapBytes, cached.info.data.size);
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout = cached.info.mip_layout;
	} else if (cached.info.resources.layers > requested.resources.layers) {
		// Extend only a layout whose byte ranges can actually be grown. Mip and
		// layer counts are independent capacities, not a lexicographic maximum;
		// importing cached mip counts would leave the requested layout invalid.
		(void)GrowImageToLayers(info, cached.info.resources.layers);
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		if (!cached.info.resources.Contains(info.resources)) {
			// A copy covers only common subresources. Seed newly added layers/mips
			// from their guest backing before retaining the existing native contents.
			InitializeImage(replacement_id);
		}
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
	// The replacement covers the same guest bytes: it takes over the watch on their pages. Freeing
	// dropped it and the replacement's first use set it again, two page-watcher passes over the whole
	// range (a 135 MiB shadow-map array switching between depth and colour: ~10 ms each).
	if (cached.IsTracked() && !replacement.IsTracked() && replacement.info.data.address == cached.info.data.address &&
	    replacement.info.data.size == cached.info.data.size) {
		replacement.track_addr      = cached.track_addr;
		replacement.track_addr_end  = cached.track_addr_end;
		replacement.untracked_holes = std::move(cached.untracked_holes);
		cached.untracked_holes.clear();
		cached.track_addr     = 0;
		cached.track_addr_end = 0;
	}
	FreeImage(cached_id, "depth-overlap");
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
	const auto current_frame = m_frame.load(std::memory_order_relaxed);
	const bool safe_to_delete =
	    current_frame - std::min(current_frame, cached.frame_accessed_last) > NumFramesBeforeRemoval;

	if (requested.data.address == cached.info.data.address) {
		const uint32_t requested_block = requested.bytes_per_block * requested.samples;
		const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
		if (requested.BlockExtent() != cached.info.BlockExtent() ||
		    requested_block != cached_block) {
			if (safe_to_delete) {
				FreeImage(cached_id, "same-address-extent");
			}
			return {merged_id};
		}

		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.data.size == cached.info.data.size &&
		    (requested.IsVolume() || cached.info.IsVolume())) {
			return {ExpandImage(requested, cached_id)};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     requested.mip_layout != cached.info.mip_layout)) {
			if (safe_to_delete) {
				FreeImage(cached_id, "same-address-layout");
			}
			return {merged_id};
		}
		// PPSA08394
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources && requested.type == cached.info.type &&
		    requested.extent.width > cached.info.extent.width &&
		    requested.extent.height >= cached.info.extent.height &&
		    requested.extent.depth >= cached.info.extent.depth &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format)) {
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
		if (requested.type == cached.info.type && !cached.info.resources.Contains(requested.resources)) {
			return {ExpandImage(requested, cached_id)};
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

	if (requested.data.address > cached.info.data.address) {
		const int32_t mip = requested.MipOf(cached.info);
		if (mip >= 0) {
			const int32_t layer = requested.SliceOf(cached.info, mip);
			if (layer >= 0) {
				return {cached_id, mip, layer};
			}
		}
		if (safe_to_delete) {
			if (SlowLog::Threshold() > 0.0 && cached.info.data.size >= PartialDirtyMinSize) {
				std::printf("[tsc %llu] OVERLAP-FREE requested=0x%llx+0x%llx fmt=%u %ux%u cached=0x%llx+0x%llx "
				            "fmt=%u %ux%u idle_frames=%llu gpu=%d target=%d\n",
				            static_cast<unsigned long long>(__rdtsc()),
				            static_cast<unsigned long long>(requested.data.address),
				            static_cast<unsigned long long>(requested.data.size),
				            static_cast<uint32_t>(requested.guest_format), requested.extent.width,
				            requested.extent.height, static_cast<unsigned long long>(cached.info.data.address),
				            static_cast<unsigned long long>(cached.info.data.size),
				            static_cast<uint32_t>(cached.info.guest_format), cached.info.extent.width,
				            cached.info.extent.height,
				            static_cast<unsigned long long>(current_frame - std::min(current_frame, cached.frame_accessed_last)),
				            cached.IsGpuModified() ? 1 : 0, cached.binding.is_target ? 1 : 0);
			}
			FreeImage(cached_id, "overlap-after");
		}
		return {};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (layer >= 0) {
			if (cached.binding.is_target) {
				cached.binding.needs_rebind = true;
				if (merged_id) {
					m_slot_images[merged_id].binding.is_target = true;
				}
				FreeImage(cached_id, "overlap-target-mip");
				return {merged_id};
			}
			if (merged_id) {
				CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
				             static_cast<uint32_t>(layer));
				FreeImage(cached_id, "overlap-copy-mip");
			}
		}
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      expanded    = m_slot_images[expanded_id];
	auto&      source      = m_slot_images[source_id];
	expanded.usage         = source.usage;
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	CopyImage(expanded_id, source_id);
	FreeImage(source_id, "expand");
	return expanded_id;
}

struct TextureCache::TextureTransferPlan {
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

struct TextureCache::DownloadPlan {
	TextureTransferPlan texture;
	bool                depth_target = false;
	bool                valid        = false;
};

TextureCache::TextureTransferPlan
TextureCache::BuildTextureTransfer(const Image& image, BindingType binding,
                                    TransferDirection direction) const {
	const auto& info             = image.info;
	const bool  upload           = direction == TransferDirection::Upload;
	const bool  render_target    = binding == BindingType::RenderTarget;
	const bool  video_out        = binding == BindingType::VideoOut;
	auto        format           = info.guest_format;
	uint32_t    layers           = info.TransferLayers();
	bool        volume           = info.IsVolume();
	bool        allow_depth_tile = upload;
	const char* owner            = "TextureCache readback";

	TextureTransferPlan plan;
	plan.swap_bgra16 = info.bgra16 && (!upload || render_target || video_out);
	if (render_target) {
		format = ImageOps::RenderTargetTransferFormat(info.bytes_per_block);
	}
	if (video_out) {
		allow_depth_tile = false;
	} else if (render_target || binding == BindingType::Storage) {
		allow_depth_tile = true;
	}
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

	plan.layout  = TextureCalcUploadLayout(format, info.extent.width, info.extent.height,
	                                       info.resources.levels, layers, info.tile_mode,
	                                       info.data.size, allow_depth_tile, volume, owner);
	plan.regions = TextureBuildImageCopies(plan.layout);
	if (info.IsDepth()) {
		for (auto& region: plan.regions) {
			region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
		}
	}
	if (plan.layout.surface.description.tile_mode != Prospero::TileMode::kLinear) {
		if (!TextureBuildGpuTileInfos(info.data.size, plan.regions, plan.layout,
		                              info.resources.levels, plan.tiles)) {
			return plan;
		}
	}
	plan.valid = true;
	return plan;
}

TextureCache::DownloadPlan TextureCache::BuildDownload(const Image& image) const {
	const auto&  info    = image.info;
	const auto   binding = UploadBinding(image);
	DownloadPlan plan {.depth_target = binding == BindingType::DepthTarget};
	if (info.samples != 1 || image.backing.samples != 1) {
		return plan;
	}
	if (plan.depth_target) {
		plan.valid = IsSupportedDepthPlaneReadback(info) && info.resources.layers != 0 &&
		             info.data.size % info.resources.layers == 0 &&
		             Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
		return plan;
	}
	if (info.metadata.compression != VideoOutCompression::Uncompressed) {
		return plan;
	}
	plan.texture = BuildTextureTransfer(image, binding, TransferDirection::Download);
	plan.valid   = plan.texture.valid;
	return plan;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	const auto& info    = image.info;
	const auto  binding = UploadBinding(image);
	const auto  upload  = [&](std::vector<vk::BufferImageCopy>& copies, TileManager::Result linear) {
		for (auto& copy: copies) {
			copy.bufferOffset += linear.offset;
		}
		image.Upload(copies, linear.buffer, linear.offset, linear.size);
	};

	if (binding != BindingType::DepthTarget) {
		const auto plan_start = std::chrono::steady_clock::now();
		auto plan = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		SlowLog::Scope slow([&](double ms) {
			std::printf("SLOW UploadImage %.1f ms (plan %.1f ms) addr=0x%llx size=0x%llx regions=%zu tiles=%zu "
			            "linear=0x%llx\n",
			            ms,
			            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - plan_start).count() - ms,
			            static_cast<unsigned long long>(info.data.address), static_cast<unsigned long long>(info.data.size),
			            plan.regions.size(), plan.tiles.size(), static_cast<unsigned long long>(plan.LinearSize()));
		}, SlowLog::HitchThreshold());
		if (!plan.valid) {
			EXIT("TextureCache: invalid texture upload: binding=%u addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 " format=%u tile=%u family=%u extent=%ux%ux%u "
			     "pitch=%u levels=%u layers=%u samples=%u\n",
			     static_cast<uint32_t>(binding), info.data.address, info.data.size,
			     static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
			     static_cast<uint32_t>(plan.layout.surface.texture.block.family), info.extent.width,
			     info.extent.height, info.extent.depth, info.pitch, info.resources.levels,
			     info.resources.layers, info.samples);
		}
		TileManager::Result linear {source.Handle(), source_offset, info.data.size};
		if (!plan.tiles.empty()) {
			linear = m_tiler.Detile(source.Handle(), source_offset, info.data.size,
			                        plan.LinearSize(), plan.tiles);
		}
		if (plan.swap_bgra16) {
			linear = m_tiler.SwapBgra16(linear);
		}
		upload(plan.regions, linear);
		return;
	}

	if (info.samples != 1 || image.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto          layers          = info.resources.layers;
	const auto          full_slice_size = info.data.size / layers;
	auto                copies          = BuildDepthCopies(info, full_slice_size);
	TileManager::Result linear {source.Handle(), source_offset, source.Size() - source_offset};
	if (info.IsTiled()) {
		const auto tiles = BuildDepthTiles(info);
		linear =
		    m_tiler.Detile(source.Handle(), source_offset, info.data.size, info.data.size, tiles);
	}
	const auto transfer_bytes = DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers);
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

void TextureCache::UploadStencil(Image& image, Buffer& source, uint64_t source_offset) {
	const auto&     info = image.info;
	TileBlockLayout block {};
	if (!info.HasStencil() || info.samples != 1 || image.backing.samples != 1 ||
	    info.resources.layers == 0 || info.stencil.size % info.resources.layers != 0 ||
	    !TileGetBlockLayout(TileBlockFamily::Depth64KB, 1, block) ||
	    (info.pixel_format != vk::Format::eD32SfloatS8Uint &&
	     info.pixel_format != vk::Format::eD24UnormS8Uint &&
	     info.pixel_format != vk::Format::eD16UnormS8Uint)) {
		EXIT("TextureCache: invalid stencil upload: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     " layers=%u samples=%u format=%u\n",
		     info.stencil.address, info.stencil.size, info.resources.layers, info.samples,
		     static_cast<uint32_t>(info.pixel_format));
	}
	const auto     layers     = info.resources.layers;
	const uint64_t slice_size = info.stencil.size / layers;
	const auto     align      = [](uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
	};
	const uint64_t pitch = align(info.extent.width, block.block_width);
	const uint64_t rows  = align(info.extent.height, block.block_height);
	if (info.tile_mode == Prospero::TileMode::kLinear || pitch * rows != slice_size ||
	    pitch > UINT32_MAX) {
		LOGF("TextureCache: stencil upload skipped addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     " extent=%ux%u pitch=%" PRIu64 " rows=%" PRIu64 " layers=%u tile=%u\n",
		     info.stencil.address, info.stencil.size, info.extent.width, info.extent.height, pitch,
		     rows, layers, static_cast<uint32_t>(info.tile_mode));
		return;
	}
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> copies(layers);
	tiles.reserve(layers);
	for (uint32_t layer = 0; layer < layers; layer++) {
		const uint64_t offset  = slice_size * layer;
		auto&          copy    = copies[layer];
		copy.bufferOffset      = offset;
		copy.bufferRowLength   = static_cast<uint32_t>(pitch);
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {vk::ImageAspectFlagBits::eStencil, 0, layer, 1};
		copy.imageExtent       = {info.extent.width, info.extent.height, 1};
		tiles.push_back({block.family, 1, offset, slice_size, offset, slice_size, 0,
		                 info.extent.width, info.extent.height, 1, static_cast<uint32_t>(pitch)});
		tiles.back().surface_z = layer;
	}
	const auto linear =
	    m_tiler.Detile(source.Handle(), source_offset, info.stencil.size, info.stencil.size, tiles);
	for (auto& copy: copies) {
		copy.bufferOffset += linear.offset;
	}
	image.Upload(copies, linear.buffer, linear.offset, linear.size);
}

// Local diagnostic: why the latest UploadImagePartial fell back to a whole upload (SLOW log).
static thread_local const char* g_partial_fail = "";
static bool PartialFail(const char* why) {
	g_partial_fail = why;
	return false;
}

void TextureCache::InitializeImage(ImageId id) {
	using Clock = std::chrono::steady_clock;
	auto&             image = m_slot_images[id];
	Clock::time_point marks[3] {Clock::now()};
	const char*       kind = image.IsPartiallyCpuDirty() ? "partial"
	                         : image.IsBufferModified()  ? "buffer"
	                         : image.IsCpuDirty()        ? "cpu"
	                                                     : "clean";
	SlowLog::Scope    slow([&](double ms) {
        const auto at = [&](int i) {
            return marks[i] == Clock::time_point {}
		               ? 0.0
		               : std::chrono::duration<double, std::milli>(marks[i] - marks[0]).count();
        };
        std::printf("SLOW InitializeImage %.1f ms addr=0x%llx size=0x%llx levels=%u layers=%u kind=%s track=%.1f "
		               "source=%.1f partial_fail=%s\n",
		               ms, static_cast<unsigned long long>(image.info.data.address),
		               static_cast<unsigned long long>(image.info.data.size), image.info.resources.levels,
		               image.info.resources.layers, kind, at(1), at(2), g_partial_fail);
    }, SlowLog::HitchThreshold());
	if (image.info.data.Empty()) {
		return;
	}
	TrackImage(id);
	marks[1] = Clock::now();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			FinishRefresh(image);
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		bool uploaded = false;
		if (image.IsPartiallyCpuDirty() && !image.IsBufferModified()) {
			g_partial_fail = "";
			uploaded = UploadImagePartial(image);
			if (!uploaded) {
				LiveCounters::Add(LiveCounters::PartialFallbacks);
			}
		}
		if (!uploaded) {
			const auto [source, source_offset] =
			    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
			marks[2] = Clock::now();
			if (source == nullptr) {
				EXIT("TextureCache: failed to obtain image upload source\n");
			}
			LiveCounters::Add(LiveCounters::FullUploads);
			LiveCounters::Add(LiveCounters::FullUploadBytes, image.info.data.size);
			static const bool log_uploads = std::getenv("KYTY_UPLOAD_LOG") != nullptr;
			if (log_uploads) {
				std::printf("[tsc %llu] UPLOAD %s addr=0x%llx size=0x%llx fmt=%u %ux%u layers=%u levels=%u tile=%u "
				            "target=%d usage=%d%d%d%d gpu=%d serial=%llu\n",
				            static_cast<unsigned long long>(__rdtsc()), kind,
				            static_cast<unsigned long long>(image.info.data.address),
				            static_cast<unsigned long long>(image.info.data.size),
				            static_cast<uint32_t>(image.info.guest_format), image.info.extent.width,
				            image.info.extent.height, image.info.resources.layers, image.info.resources.levels,
				            static_cast<uint32_t>(image.info.tile_mode), image.binding.is_target ? 1 : 0,
				            image.usage.texture ? 1 : 0, image.usage.storage ? 1 : 0,
				            image.usage.render_target ? 1 : 0, image.usage.depth_target ? 1 : 0,
				            image.IsGpuModified() ? 1 : 0, static_cast<unsigned long long>(image.serial));
			}
			UploadImage(image, *source, source_offset);
			if (kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 2 &&
			    PartialDirtyCandidate(image)) {
				UpdatePartialHashes(image, true);
			}
		}
		image.ClearBufferModified();
	}
	if (image.IsCpuDirty()) {
		FinishRefresh(image);
	}
}

namespace {

[[nodiscard]] bool IntersectsRanges(const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                                    uint64_t begin, uint64_t end) {
	const auto at = std::upper_bound(ranges.begin(), ranges.end(), begin,
	                                 [](uint64_t value, const auto& range) { return value < range.second; });
	return at != ranges.end() && at->first < end;
}

// KYTY_PARTIAL_ROW_BANDS: the game streams into small parts of large textures (mip 0 of an 8192x8192
// BC texture is 64 MiB of an 85 MiB image), and a refresh that picks whole subresources then
// uploads more than half the image, i.e. all of it (three times within 60 ms: a 58 ms frame).
// Within a subresource, blocks are laid out row-major (block index = by * blocks_per_row + bx) and
// these families swizzle inside a block with in-block coordinate bits only, so the rows of blocks
// over the dirty ranges detile like a shorter surface of the same width (no shader change).
// Render-target and depth blocks XOR the block row's parity into the in-block offset: whole.
[[nodiscard]] bool RowBandable(const GpuTileInfo& tile) {
	if (tile.tail || tile.depth != 1) {
		return false;
	}
	switch (tile.family) {
		case TileBlockFamily::Standard256B:
		case TileBlockFamily::Standard4KB:
		case TileBlockFamily::Standard64KB:
		case TileBlockFamily::Prt64KB: return true;
		default: return false;
	}
}

struct RowBand {
	uint32_t first = 0; // block rows [first, last)
	uint32_t last  = 0;
};

// The block rows of one subresource [begin, end) (rows of row_bytes) that dirty ranges touch.
void DirtyRowBands(const std::vector<std::pair<uint64_t, uint64_t>>& dirty, uint64_t begin, uint64_t end,
                   uint64_t row_bytes, uint32_t rows, std::vector<RowBand>& bands) {
	bands.clear();
	auto at = std::upper_bound(dirty.begin(), dirty.end(), begin,
	                           [](uint64_t value, const auto& range) { return value < range.second; });
	for (; at != dirty.end() && at->first < end; ++at) {
		const uint64_t lo    = std::max(at->first, begin) - begin;
		const uint64_t hi    = std::min(at->second, end) - begin;
		const auto     first = static_cast<uint32_t>(lo / row_bytes);
		const auto     last  = static_cast<uint32_t>(std::min<uint64_t>(rows, (hi + row_bytes - 1) / row_bytes));
		if (!bands.empty() && first <= bands.back().last) {
			bands.back().last = std::max(bands.back().last, last);
		} else {
			bands.push_back({first, last});
		}
	}
}

[[nodiscard]] uint64_t HashGuestRange(uint64_t vaddr, uint64_t size) {
	if (const auto* data = LibKernel::Memory::TryGetBackingPointer(vaddr, size)) {
		return XXH3_64bits(data, static_cast<size_t>(size));
	}
	thread_local std::vector<uint8_t> bytes;
	bytes.resize(size);
	if (!LibKernel::Memory::TryReadBacking(vaddr, bytes.data(), size) &&
	    !LibKernel::Memory::TryReadPrtBacking(vaddr, bytes.data(), size)) {
		// Unreadable (unbacked sparse pages): equal to itself until the pages become readable.
		return 0x5ca1ab1e0ddba11ull;
	}
	return XXH3_64bits(bytes.data(), static_cast<size_t>(size));
}

} // namespace

// KYTY_PARTIAL_IMAGE_DIRTY=2: every granule of a large image is hashed at a whole upload, and the
// dirty granules again after a partial one; a partial upload first checks that the granules it
// leaves alone still hold the data uploaded last time (a write the page watchers missed).
void TextureCache::UpdatePartialHashes(Image& image, bool all) {
	const auto& info  = image.info;
	const auto  count = (info.data.size + PartialDirtyGranule - 1) / PartialDirtyGranule;
	if (image.partial_hashes.size() != count) {
		if (!all) {
			return;
		}
		image.partial_hashes.assign(count, 0);
	}
	for (uint64_t index = 0; index < count; index++) {
		const auto begin = info.data.address + index * PartialDirtyGranule;
		const auto size  = std::min(PartialDirtyGranule, info.data.End() - begin);
		if (all || IntersectsRanges(image.CpuDirtyRanges(), begin, begin + size)) {
			image.partial_hashes[index] = HashGuestRange(begin, size);
		}
	}
}

bool TextureCache::CheckPartialHashes(const Image& image) {
	const auto& info  = image.info;
	const auto  count = (info.data.size + PartialDirtyGranule - 1) / PartialDirtyGranule;
	if (image.partial_hashes.size() != count) {
		std::printf("PARTIAL VERIFY no baseline addr=0x%llx size=0x%llx\n",
		            static_cast<unsigned long long>(info.data.address),
		            static_cast<unsigned long long>(info.data.size));
		return false;
	}
	uint64_t mismatches = 0;
	uint64_t first      = 0;
	for (uint64_t index = 0; index < count; index++) {
		const auto begin = info.data.address + index * PartialDirtyGranule;
		const auto size  = std::min(PartialDirtyGranule, info.data.End() - begin);
		if (!IntersectsRanges(image.CpuDirtyRanges(), begin, begin + size) &&
		    HashGuestRange(begin, size) != image.partial_hashes[index]) {
			if (mismatches++ == 0) {
				first = begin;
			}
		}
	}
	if (mismatches != 0) {
		std::printf("PARTIAL VERIFY mismatch addr=0x%llx size=0x%llx granules=%llu first=0x%llx "
		            "dirty_ranges=%zu\n",
		            static_cast<unsigned long long>(info.data.address),
		            static_cast<unsigned long long>(info.data.size),
		            static_cast<unsigned long long>(mismatches), static_cast<unsigned long long>(first),
		            image.CpuDirtyRanges().size());
		return false;
	}
	return true;
}

// A depth array whose CPU writes cover some layers (the game streams shadow maps into an 80-layer
// pool, a few 4 MiB layers at a time): only those layers are staged, detiled and copied. The
// whole-image path staged and detiled every layer (320 MiB), 20-65 ms per refresh.
bool TextureCache::UploadDepthPartial(Image& image) {
	const auto& info = image.info;
	// What UploadImage's depth path handles without a D16 promotion, one level, whole layers.
	if (info.samples != 1 || image.backing.samples != 1 || info.HasStencil() || info.IsVolume() ||
	    info.resources.levels != 1 || info.resources.layers < 2 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block ||
	    DepthAspectTransferBytes(info.pixel_format) != info.bytes_per_block) {
		return PartialFail("depth_kind");
	}
	const uint32_t layers = info.resources.layers;
	const uint64_t slice  = info.data.size / layers;
	const uint64_t base   = info.data.address;
	const auto&    dirty  = image.CpuDirtyRanges();
	std::vector<uint32_t> picked;
	for (uint32_t layer = 0; layer < layers; ++layer) {
		if (IntersectsRanges(dirty, base + slice * layer, base + slice * (layer + 1))) {
			picked.push_back(layer);
		}
	}
	if (picked.empty()) {
		LiveCounters::Add(LiveCounters::PartialUploads);
		return true;
	}
	if (slice * picked.size() > info.data.size / PartialUploadMaxShare) {
		return PartialFail("depth_share");
	}
	// Runs of consecutive layers stage as one piece each.
	std::vector<BufferCache::StagingPiece> pieces;
	std::vector<uint64_t>                  staged(picked.size());
	uint64_t                               total = 0;
	for (size_t i = 0; i < picked.size(); ++i) {
		const uint64_t vaddr = base + slice * picked[i];
		if (!pieces.empty() && pieces.back().vaddr + pieces.back().size == vaddr) {
			staged[i] = pieces.back().offset + pieces.back().size;
			pieces.back().size += slice;
		} else {
			const uint64_t offset = (total + 255) & ~uint64_t {255};
			pieces.push_back({vaddr, slice, offset});
			staged[i] = offset;
		}
		total = pieces.back().offset + pieces.back().size;
	}
	const auto [source, source_offset] = m_buffer_cache.StageImagePieces(pieces, total);
	if (source == nullptr) {
		return PartialFail("depth_stage");
	}
	// The picked layers, packed: as the whole path's copies and tiles, at their own layer.
	std::vector<vk::BufferImageCopy> copies(picked.size());
	for (size_t i = 0; i < picked.size(); ++i) {
		auto& copy             = copies[i];
		copy.bufferOffset      = info.IsTiled() ? slice * i : staged[i];
		copy.bufferRowLength   = info.pitch;
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, picked[i], 1};
		copy.imageExtent       = {info.extent.width, info.extent.height, 1};
	}
	TileManager::Result linear {source->Handle(), source_offset, total};
	if (info.IsTiled()) {
		TileBlockLayout block {};
		EXIT_NOT_IMPLEMENTED(!TileGetBlockLayout(TileBlockFamily::Depth64KB, info.bytes_per_block, block));
		std::vector<GpuTileInfo> tiles;
		tiles.reserve(picked.size());
		for (size_t i = 0; i < picked.size(); ++i) {
			tiles.push_back({block.family, block.bytes_per_element, slice * i, slice, staged[i], slice, 0,
			                 info.extent.width, info.extent.height, 1, info.pitch});
			tiles.back().surface_z = picked[i];
		}
		linear = m_tiler.Detile(source->Handle(), source_offset, total, slice * picked.size(), tiles);
	}
	for (auto& copy: copies) {
		copy.bufferOffset += linear.offset;
	}
	image.Upload(copies, linear.buffer, linear.offset, linear.size);
	LiveCounters::Add(LiveCounters::PartialUploads);
	LiveCounters::Add(LiveCounters::PartialUploadBytes, total);
	return true;
}

bool TextureCache::UploadImagePartial(Image& image) {
	const auto& info    = image.info;
	const auto  binding = UploadBinding(image);
	const auto  base    = info.data.address;
	if (binding == BindingType::DepthTarget) {
		return UploadDepthPartial(image);
	}
	if (info.IsVolume() || base % 256 != 0) {
		return PartialFail("binding");
	}
	auto& cached = m_partial_plans[image.serial];
	if (cached.plan == nullptr || cached.binding != binding) {
		cached.plan    = std::make_shared<TextureTransferPlan>(
            BuildTextureTransfer(image, binding, TransferDirection::Upload));
		cached.binding = binding;
	}
	const auto& plan = *cached.plan;
	if (!plan.valid || plan.swap_bgra16 || plan.tiles.empty() ||
	    plan.tiles.size() != plan.regions.size()) {
		return PartialFail("plan");
	}
	if (kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 2 &&
	    !CheckPartialHashes(image)) {
		return PartialFail("hashes");
	}
	// The subresources over the dirty ranges (region i is the copy of tile i), or the rows of
	// blocks over them (KYTY_PARTIAL_ROW_BANDS).
	struct Pick {
		uint64_t begin     = 0;
		uint64_t end       = 0;
		size_t   index     = 0;
		uint32_t first_row = 0; // a band's block rows; last_row 0: the whole subresource
		uint32_t last_row  = 0;
	};
	// A fragmented dirty set uploads the whole subresource rather than many small dispatches.
	constexpr size_t MaxBandsPerSubresource = 16;
	const auto&          dirty    = image.CpuDirtyRanges();
	const bool           bands_on = kyty_local_partial_row_bands_mode.load(std::memory_order_relaxed) != 0;
	uint64_t             picked   = 0;
	std::vector<Pick>    picks;
	std::vector<RowBand> bands;
	for (size_t index = 0; index < plan.tiles.size(); index++) {
		const auto& tile  = plan.tiles[index];
		const auto  begin = base + tile.tiled_offset;
		const auto  end   = begin + tile.tiled_size;
		if (plan.regions[index].bufferOffset < tile.linear_offset) {
			return PartialFail("tile_offset");
		}
		if (!IntersectsRanges(dirty, begin, end)) {
			continue;
		}
		TileBlockLayout block {};
		if (bands_on && RowBandable(tile) && TileGetBlockLayout(tile.family, tile.bytes_per_element, block) &&
		    block.block_width != 0 && block.block_height != 0 && tile.tiled_width % block.block_width == 0 &&
		    tile.tiled_height % block.block_height == 0) {
			const uint64_t row_bytes = uint64_t {tile.tiled_width / block.block_width} * block.block_size;
			const uint32_t rows      = tile.tiled_height / block.block_height;
			if (rows > 1 && row_bytes * rows == tile.tiled_size) {
				DirtyRowBands(dirty, begin, end, row_bytes, rows, bands);
				if (bands.size() <= MaxBandsPerSubresource) {
					for (const auto& band: bands) {
						// Rows past the subresource's last element row are padding only.
						if (uint64_t {band.first} * block.block_height >= tile.height) {
							continue;
						}
						picks.push_back({begin + band.first * row_bytes, begin + band.last * row_bytes, index,
						                 band.first, band.last});
						picked += uint64_t {band.last - band.first} * row_bytes;
					}
					continue;
				}
			}
		}
		picks.push_back({begin, end, index});
		picked += tile.tiled_size;
	}
	if (picked > info.data.size / PartialUploadMaxShare) {
		return PartialFail("share");
	}
	if (picks.empty()) {
		// Only padding between subresources changed.
		LiveCounters::Add(LiveCounters::PartialUploads);
		return true;
	}
	// Stage the picked guest ranges back to back; runs that nearly touch stay one piece.
	std::sort(picks.begin(), picks.end(), [](const Pick& left, const Pick& right) {
		return left.begin < right.begin;
	});
	std::vector<BufferCache::StagingPiece> pieces;
	std::vector<size_t>                    piece_of(picks.size());
	uint64_t                               total = 0;
	for (size_t at = 0; at < picks.size(); at++) {
		const auto begin = picks[at].begin & ~uint64_t {255};
		const auto end   = std::min(info.data.End(), (picks[at].end + 255) & ~uint64_t {255});
		if (!pieces.empty() && begin <= pieces.back().vaddr + pieces.back().size + PartialStageGap) {
			auto& piece = pieces.back();
			piece.size  = std::max(piece.vaddr + piece.size, end) - piece.vaddr;
			total       = piece.offset + piece.size;
		} else {
			const auto offset = (total + 255) & ~uint64_t {255};
			pieces.push_back({begin, end - begin, offset});
			total = offset + (end - begin);
		}
		piece_of[at] = pieces.size() - 1;
	}
	// The picked tiles read from their piece and detile into a packed linear buffer.
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> regions;
	tiles.reserve(picks.size());
	regions.reserve(picks.size());
	uint64_t linear_total = 0;
	for (size_t at = 0; at < picks.size(); at++) {
		const auto& piece  = pieces[piece_of[at]];
		auto        tile   = plan.tiles[picks[at].index];
		auto        region = plan.regions[picks[at].index];
		if (picks[at].last_row != 0) {
			// A band: a shorter surface of the same width starting at its first row of blocks.
			TileBlockLayout block {};
			(void)TileGetBlockLayout(tile.family, tile.bytes_per_element, block);
			const uint32_t texel_height = plan.layout.surface.texture.texel_height;
			const uint32_t block_rows   = picks[at].last_row - picks[at].first_row;
			const uint32_t first        = picks[at].first_row * block.block_height; // element row
			const uint32_t count        = std::min(block_rows * block.block_height, tile.height - first);
			tile.height                 = count;
			tile.tiled_height           = block_rows * block.block_height;
			tile.tiled_size             = picks[at].end - picks[at].begin;
			tile.linear_size            = uint64_t {tile.pitch} * count * tile.bytes_per_element;
			region.imageOffset.y        = static_cast<int32_t>(first * texel_height);
			region.imageExtent.height   = std::min(count * texel_height, region.imageExtent.height - first * texel_height);
		}
		const auto linear_offset = (linear_total + 255) & ~uint64_t {255};
		region.bufferOffset       = linear_offset + (region.bufferOffset - tile.linear_offset);
		tile.linear_offset        = linear_offset;
		tile.tiled_offset         = piece.offset + (picks[at].begin - piece.vaddr);
		linear_total              = linear_offset + tile.linear_size;
		tiles.push_back(tile);
		regions.push_back(region);
	}
	const auto [source, source_offset] = m_buffer_cache.StageImagePieces(pieces, total);
	if (source == nullptr) {
		return PartialFail("stage");
	}
	const auto linear = m_tiler.Detile(source->Handle(), source_offset, total, linear_total, tiles);
	for (auto& region: regions) {
		region.bufferOffset += linear.offset;
	}
	image.Upload(regions, linear.buffer, linear.offset, linear.size);
	LiveCounters::Add(LiveCounters::PartialUploads);
	LiveCounters::Add(LiveCounters::PartialUploadBytes, total);
	if (kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 2) {
		UpdatePartialHashes(image, false);
	}
	return true;
}

void TextureCache::PrepareDccClear(ImageId id, const ImageDesc& desc) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc) {
		return;
	}
	auto&      image       = m_slot_images[id];
	bool       changed     = !(image.info.metadata == desc.info.metadata);
	image.info.metadata    = desc.info.metadata;
	auto [entry, inserted] = m_surface_metas.try_emplace(
	    desc.info.metadata.range.address, MetaDataInfo {.type = MetaDataInfo::Type::Dcc});
	MetaFilterAdd(desc.info.metadata.range.address);
	changed |= inserted;
	auto& metadata = entry->second;
	if (!inserted && metadata.type == MetaDataInfo::Type::PendingDcc) {
		metadata.type = MetaDataInfo::Type::Dcc;
		changed       = true;
	} else if (metadata.type != MetaDataInfo::Type::Dcc) {
		EXIT("TextureCache: image reuses non-DCC metadata\n");
	}
	// (Only when the record changed: every DCC target acquisition passes here.)
	if (changed) m_meta_epoch.fetch_add(1, std::memory_order_release);
	if (metadata.clear_mask == 0 || image.info.resources.levels != 1 ||
	    desc.info.metadata.range.size == 0 || metadata.fill_size < desc.info.metadata.range.size) {
		return;
	}
	vk::ClearValue clear {};
	if (!DecodeDccClear(desc, image.backing.format, metadata.fill_value, clear.color)) {
		return;
	}
	const auto& view           = desc.view_info;
	const bool  volume_texture = image.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : view.base_layer;
	const auto  count = volume_texture ? std::max(image.info.extent.depth >> view.base_level, 1u)
	                                   : view.layer_count;
	if (first >= 32 || count > 32 - first) {
		return;
	}
	// The metadata fill covers the complete allocation. Consume each layer only after its
	// native image contents exist; already materialized layers may have been rendered since.
	for (uint32_t layer = first; layer < first + count;) {
		if ((metadata.clear_mask & (1u << layer)) == 0) {
			layer++;
			continue;
		}
		const auto start = layer;
		uint32_t   mask  = 0;
		do {
			mask |= 1u << layer++;
		} while (layer < first + count && (metadata.clear_mask & (1u << layer)) != 0);
		ClearImage(m_scheduler.Current(), id,
		           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count, start,
		            layer - start},
		           clear);
		metadata.clear_mask &= ~mask;
	}
}

void TextureCache::RefreshImage(ImageId id) {
	TrackImage(id);
	auto& image = m_slot_images[id];
	if (image.IsStencilModified()) {
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.stencil.address, image.info.stencil.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain stencil upload source\n");
		}
		UploadStencil(image, *source, source_offset);
		image.ClearStencilModified();
	}
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

void TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
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
		if (owner != nullptr && owner->info.data.address == stencil.address) {
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
	if (std::find(m_stencil_associations.begin(), m_stencil_associations.end(), association) ==
	    m_stencil_associations.end()) {
		m_stencil_associations.push_back(association);
	}
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	size_t         slow_candidates = 0;
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW FindImage %.1f ms addr=0x%llx size=0x%llx type=%u fmt=%u %ux%ux%u levels=%u layers=%u "
		            "tile=%u candidates=%zu\n",
		            ms, static_cast<unsigned long long>(desc.info.data.address),
		            static_cast<unsigned long long>(desc.info.data.size), static_cast<uint32_t>(desc.type),
		            static_cast<uint32_t>(desc.info.guest_format), desc.info.extent.width, desc.info.extent.height,
		            desc.info.extent.depth, desc.info.resources.levels, desc.info.resources.layers,
		            static_cast<uint32_t>(desc.info.tile_mode), slow_candidates);
	});
	if (desc.info.data.size) m_buffer_cache.DrainGuestReadback(desc.info.data.address, desc.info.data.size);
	if (desc.info.stencil.size) m_buffer_cache.DrainGuestReadback(desc.info.stencil.address, desc.info.stencil.size);
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		ImageIds candidates;
		if (kyty_local_preparation_lookup_mode.load(std::memory_order_relaxed) != 0) {
			// Every exact match starts on the requested first page. Traversing
			// that page preserves the original order (including the last match)
			// without visiting and deduplicating the same owners on later pages.
			ImagePageTable::PageRange pages {};
			if (ImagePageTable::TryGetPageRange(desc.info.data.address, desc.info.data.size, pages)) {
				if (const auto* owners = m_image_page_table.Find(pages.first)) {
					owners->ForEach([&](ImageId id) {
						const auto* image = m_slot_images.try_get(id);
						if (image != nullptr && SameBacking(image->info, desc.info, exact_format)) {
							result = id;
						}
					});
				}
			}
		}
		if (!result) {
			candidates      = FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
			slow_candidates = candidates.size();
			for (const auto id: candidates) {
				const auto& image = m_slot_images[id];
				if (SameBacking(image.info, desc.info, exact_format)) {
					result = id;
				}
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
			} else if (!resolved.info.resources.Contains(desc.info.resources)) {
				if (SlowLog::Threshold() > 0.0 && resolved.info.data.size >= PartialDirtyMinSize) {
					std::printf("[tsc %llu] FIND-RESOURCES requested=0x%llx+0x%llx fmt=%u %ux%u layers=%u levels=%u "
					            "type=%u tile=%u\n",
					            static_cast<unsigned long long>(__rdtsc()),
					            static_cast<unsigned long long>(desc.info.data.address),
					            static_cast<unsigned long long>(desc.info.data.size),
					            static_cast<uint32_t>(desc.info.guest_format), desc.info.extent.width,
					            desc.info.extent.height, desc.info.resources.layers, desc.info.resources.levels,
					            static_cast<uint32_t>(desc.type), static_cast<uint32_t>(desc.info.tile_mode));
				}
				FreeImage(result, "find-resources");
				result = {};
			}
		}
		if (!result) {
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
		}
		auto& image = m_slot_images[result];
		if (desc.info.tile_mode == Prospero::TileMode::kDepth &&
		    (desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
		    (image.info.data.size != desc.info.data.size ||
		     !(image.info.resources == desc.info.resources))) {
			LOGF("TextureCache: depth-tiled view matched a differently shaped image: requested"
			     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " levels=%u layers=%u extent=%ux%u,"
			     " image addr=0x%016" PRIx64 " size=0x%016" PRIx64 " levels=%u layers=%u"
			     " extent=%ux%u fmt=%u depth=%d usage=%s%s%s%s gpu_modified=%d cpu_dirty=%d"
			     " view_mip=%d view_layer=%d\n",
			     desc.info.data.address, desc.info.data.size, desc.info.resources.levels,
			     desc.info.resources.layers, desc.info.extent.width, desc.info.extent.height,
			     image.info.data.address, image.info.data.size, image.info.resources.levels,
			     image.info.resources.layers, image.info.extent.width, image.info.extent.height,
			     static_cast<uint32_t>(image.info.guest_format), image.info.IsDepth() ? 1 : 0,
			     image.usage.texture ? "texture " : "", image.usage.storage ? "storage " : "",
			     image.usage.render_target ? "render_target " : "",
			     image.usage.depth_target ? "depth_target " : "", image.IsGpuModified() ? 1 : 0,
			     image.IsCpuDirty() ? 1 : 0, view_mip, view_layer);
		}
		if (desc.type == BindingType::VideoOut &&
		    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
			const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
			const bool native_current =
			    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
			if (!native_current) {
				EXIT("TextureCache: compressed video-out read requires clean native GPU "
				     "contents\n");
			}
		}
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.frame_accessed_last = m_frame.load(std::memory_order_relaxed);
		TouchImage(image);
	}
	return result;
}

bool TextureCache::SameDccSurface(const Image& image, const ImageMetadataInfo& metadata) const {
	const auto& held = image.info.metadata;
	if (held.kind != metadata.kind || !(held.range == metadata.range) || held.compression != metadata.compression) {
		return false;
	}
	const auto entry = m_surface_metas.find(metadata.range.address);
	return entry != m_surface_metas.end() && entry->second.type == MetaDataInfo::Type::Dcc;
}

bool TextureCache::IsSampledImageCurrent(ImageId id, const ImageDesc& desc, bool storage_dcc) {
	std::scoped_lock lock {m_lock};
	// A DCC image keeps the surface FindTexture gave it (its user checks for a pending clear): sampled, or written as
	// storage by a caller that checks at every use too.
	const bool dcc = (desc.type == BindingType::Texture || (storage_dcc && desc.type == BindingType::Storage)) &&
	                 desc.info.metadata.kind == ImageMetadataKind::Dcc;
	if (m_scheduler.Current().IsInvalid() || (desc.type != BindingType::Texture && desc.type != BindingType::Storage) ||
	    desc.info.data.Empty() || desc.info.IsDepth() || (desc.info.HasMetadata() && !dcc) ||
	    desc.info.HasStencil() || desc.info.tile_mode == Prospero::TileMode::kDepth) {
		return false;
	}
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered || image->depth_id ||
	    image->binding.needs_rebind || image->info.IsDepth() || (image->info.HasMetadata() && !dcc) ||
	    (dcc && !SameDccSurface(*image, desc.info.metadata)) ||
	    image->info.HasStencil() || !SameBacking(image->info, desc.info, true) ||
	    !(image->info.resources == desc.info.resources)) {
		return false;
	}
	if (Spec::Current() == nullptr) image->frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	TouchImage(*image);
	return true;
}

bool TextureCache::IsSampledDepthCurrent(ImageId id, const ImageDesc& desc, uint64_t epoch) {
	std::scoped_lock lock {m_lock};
	// (A DCC view would ask PrepareDccClear for work.)
	auto* image = m_slot_images.try_get(id);
	if (m_scheduler.Current().IsInvalid() || desc.type != BindingType::Texture || desc.info.data.Empty() ||
	    desc.info.metadata.kind == ImageMetadataKind::Dcc || image == nullptr || !image->registered || image->depth_id ||
	    image->binding.needs_rebind || !image->info.IsDepth() || image->IsCpuDirty() || image->IsBufferModified() ||
	    image->IsStencilModified() || RegistrationsSince(desc.info.data.address, desc.info.data.size, epoch)) {
		return false;
	}
	// A view of the stencil plane: the lookup finds the plane's association (one per address: AssociateStencil reuses
	// it), which leads to the depth image the last depth target binding with that plane was given, without registering
	// anything: the same while the one association there leads to this image.
	if (image->info.data.address != desc.info.data.address) {
		if (!image->info.HasStencil() || image->info.stencil.address != desc.info.data.address) return false;
		uint32_t associations = 0;
		bool     leads_here   = false;
		for (const auto association: m_stencil_associations) {
			const auto* record = m_slot_images.try_get(association);
			if (record == nullptr || !record->registered || record->info.data.address != desc.info.data.address) continue;
			++associations;
			leads_here = record->depth_id == id;
		}
		if (associations != 1 || !leads_here) return false;
	}
	if (Spec::Current() == nullptr) image->frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	TouchImage(*image);
	return true;
}

bool TextureCache::IsTargetCurrent(ImageId id, const ImageInfo& requested, uint64_t& epoch) {
	std::scoped_lock lock {m_lock};
	const auto current = m_resolution_epoch.load(std::memory_order_relaxed);
	// FindImage takes the last owner with the requested backing (on the first page or in the region);
	// only without one does it resolve an overlap (a target found as part of another image).
	if (epoch != current && RegistrationsSince(requested.data.address, 1, epoch)) {
		const auto* target = m_slot_images.try_get(id);
		ImageId     last {};
		const auto  same = [&](ImageId owner) {
			const auto* image = m_slot_images.try_get(owner);
			if (image != nullptr && SameBacking(image->info, requested, false)) last = owner;
		};
		ImagePageTable::PageRange pages {};
		if (kyty_local_preparation_lookup_mode.load(std::memory_order_relaxed) != 0 &&
		    ImagePageTable::TryGetPageRange(requested.data.address, requested.data.size, pages)) {
			if (const auto* owners = m_image_page_table.Find(pages.first)) owners->ForEach(same);
		} else {
			for (const auto owner: FindImagesInRegion(requested.data.address, requested.data.size, false)) same(owner);
		}
		if (target == nullptr || last != (SameBacking(target->info, requested, false) ? id : ImageId {})) return false;
	}
	epoch = current;
	return true;
}

bool TextureCache::TryReuseSampledImage(ImageId id, const ImageDesc& desc, uint64_t& epoch) {
	std::scoped_lock lock {m_lock};
	const auto current = m_resolution_epoch.load(std::memory_order_relaxed);
	const bool epoch_holds =
	    epoch == current ||
	    (kyty_local_texture_resolve_pages_mode.load(std::memory_order_relaxed) != 0 && epoch != 0 &&
	     !RegistrationsSince(desc.info.data.address, desc.info.data.size, epoch));
	if (m_scheduler.Current().IsInvalid() || epoch == 0 || !epoch_holds ||
	    desc.type != BindingType::Texture || desc.info.data.Empty() || desc.info.IsDepth() ||
	    desc.info.HasMetadata() || desc.info.HasStencil() ||
	    desc.info.tile_mode == Prospero::TileMode::kDepth) {
		return false;
	}
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered || image->depth_id ||
	    image->binding.needs_rebind || image->info.IsDepth() || image->info.HasMetadata() ||
	    image->info.HasStencil() || !SameBacking(image->info, desc.info, true) ||
	    !(image->info.resources == desc.info.resources)) {
		return false;
	}
	// The caller recorded this owner after an ordinary lookup, without an intervening
	// registration change. A new compatible alias also changes the epoch, so it cannot
	// silently take precedence over the cached owner. Subresource/format remaps fall back.
	if (Spec::Current() == nullptr) image->frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	TouchImage(*image);
	// Nothing registered over the pages since `epoch`, so nothing since `current` either: the
	// caller's proof moves to the current epoch (registrations stamp their pages under m_lock),
	// and the next check compares one number instead of walking the pages again.
	epoch = current;
	return true;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

bool TextureCache::HasImageStartingAt(uint64_t address) {
	std::scoped_lock lock {m_lock};
	return m_image_starts.contains(address);
}

bool TextureCache::NewImageStartsSince(uint64_t epoch, std::vector<std::pair<uint64_t, uint64_t>>& out) {
	std::scoped_lock lock {m_lock};
	// Entries carry consecutive epochs; the log covers (front - 1, current].
	if (!m_start_log.empty() && epoch + 1 < m_start_log.front().first) return false;
	const auto first = std::ranges::upper_bound(m_start_log, epoch, {}, &std::pair<uint64_t, uint64_t>::first);
	out.insert(out.end(), first, m_start_log.end());
	return true;
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	if (kyty_local_image_granules_mode.load(std::memory_order_relaxed) == 2 && !MayHaveImages(address, 1)) {
		return {};
	}
	// Only registered images starting at `address` qualify.
	if (!m_image_starts.contains(address)) {
		return {};
	}
	ImageIds matches;
	// Each qualifying image covers that byte: the images over it are the same
	// candidates, in the same order, as the images over the whole range (which
	// can span many image pages).
	for (const auto id: FindImagesInRegion(address, 1, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = m_slot_images.try_get(owner->depth_id);
		}
		if (owner == nullptr || (ensure_valid && !SafeToDownload(*owner))) {
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
	SlowLog::Scope   slow([&](double ms) {
        std::printf("SLOW FindTexture %.1f ms addr=0x%llx size=0x%llx fmt=%u %ux%ux%u levels=%u layers=%u tile=%u "
	                  "maybe_cpu_dirty=%d cpu_dirty=%d buffer_modified=%d views=%zu\n",
	                  ms, static_cast<unsigned long long>(image.info.data.address),
	                  static_cast<unsigned long long>(image.info.data.size), static_cast<uint32_t>(image.info.guest_format),
	                  image.info.extent.width, image.info.extent.height, image.info.extent.depth,
	                  image.info.resources.levels, image.info.resources.layers,
	                  static_cast<uint32_t>(image.info.tile_mode), image.IsMaybeCpuDirty() ? 1 : 0,
	                  image.IsCpuDirty() ? 1 : 0, image.IsBufferModified() ? 1 : 0, image.views.size());
    });
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
		PrepareDccClear(id, desc);
		RefreshImage(id);
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				if (!image.registered || image.depth_id) {
					EXIT("TextureCache: cannot acquire an unavailable storage image\n");
				}
				CommitGpuWrite(image);
			}
			TrackImageDownload(id, image);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	const auto view = image.FindView(desc.view_info);
	NameImageBinding(m_graphics, image, view, desc.type, desc.view_info);
	return view;
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
	PrepareDccClear(id, desc);
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	const auto view = image.FindView(desc.view_info);
	NameImageBinding(m_graphics, image, view, desc.type, desc.view_info);
	return view;
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
	RefreshImage(id);
	if (desc.info.HasMetadata()) {
		// The epoch moves only when metadata state changes: every depth target
		// acquisition passes here.
		bool changed        = !(image.info.metadata == desc.info.metadata);
		image.info.metadata = desc.info.metadata;
		auto [metadata, inserted] =
		    m_surface_metas.try_emplace(desc.info.metadata.range.address,
		                                MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                                              .clear_mask = image.info.htile_clear_mask});
		MetaFilterAdd(desc.info.metadata.range.address);
		changed |= inserted;
		if (!inserted && metadata->second.type != MetaDataInfo::Type::HTile) {
			// PS5 allocations can reuse DCC storage as HTile while the old color image is cached.
			// The depth binding defines the new type; incompatible fill state cannot carry over.
			metadata->second = {.type       = MetaDataInfo::Type::HTile,
			                    .clear_mask = image.info.htile_clear_mask};
			changed          = true;
		}
		if (changed) m_meta_epoch.fetch_add(1, std::memory_order_release);
	}
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		AssociateStencil(id, desc.info.stencil);
	}
	const auto view = image.FindView(desc.view_info);
	NameImageBinding(m_graphics, image, view, desc.type, desc.view_info);
	return view;
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (image.depth_id || image.backing.image == nullptr) {
		EXIT("TextureCache: stencil association cannot own image contents\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty()) {
		FinishRefresh(image);
	}
	image.MarkGpuModified();
	Spec::NoteGpuWrite(image.info.data.address, image.info.data.size);
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
		if (!DecodeFilledColorClear(image.info.pixel_format, packed_clear, clear.color)) {
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
	ClearImage(command, selected,
	           {aspect, 0, image.info.resources.levels, 0, image.info.TransferLayers()}, clear);
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id,
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
	const bool full_image = range.aspectMask == aspects && range.baseMipLevel == 0 &&
	                        range.levelCount == image.info.resources.levels &&
	                        range.baseArrayLayer == 0 && range.layerCount == layers;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	command.EndRendering();
	if (image.info.IsVolume() && !full_image) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		ImageViewInfo view {};
		view.format = image.backing.format;
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
	auto copies = BuildDepthCopies(info, full_slice_size);
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
	auto       guest_linear = tiled ? m_tiler.GetScratchBuffer(info.data.size)
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

void TextureCache::DownloadImageData(Image& image, Buffer& destination, uint64_t destination_offset,
                                     uint64_t destination_size, DownloadPlan plan) {
	if (!plan.valid) {
		EXIT("TextureCache: invalid image download plan\n");
	}
	destination.written_serial = m_scheduler.CommandSerial();
	if (plan.depth_target) {
		if (destination_size != image.info.data.size) {
			EXIT("TextureCache: partial depth image download is unsupported\n");
		}
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	auto&      texture   = plan.texture;
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
	if (!m_texture_cache.SafeToDownload(image)) {
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
	auto plan = m_texture_cache.BuildDownload(image);
	if (!plan.valid) {
		return false;
	}
	if (plan.depth_target && copy_size != image.info.data.size) {
		return false;
	}
	if (!plan.depth_target && levels < image.info.resources.levels) {
		auto& texture = plan.texture;
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
	InvalidateCopyFeedback(image.info.data.address, copy_size);
	NoteGpuWrite(image.info.data.address, copy_size);
	m_texture_cache.DownloadImageData(image, buffer, buf_offset, copy_size, std::move(plan));
	return true;
}

bool TextureCache::TryDownloadImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	auto plan = BuildDownload(image);
	if (!plan.valid || !SafeToDownload(image)) {
		return false;
	}
	const auto range    = image.info.data;
	auto&      download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	auto [mapped, offset] =
	    download.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	if (mapped == nullptr) {
		EXIT("TextureCache: failed to map reusable download buffer\n");
	}
	download.Commit();
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	download.Flush(offset, range.size);

	DownloadImageData(image, download, offset, range.size, std::move(plan));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	const auto tick = m_scheduler.CurrentTick();
	{
		std::lock_guard lock(m_download_mutex);
		m_pending_downloads.push_back({range.address, range.size, tick});
	}
	m_scheduler.DeferPriorityOperation([this, &download, range, mapped, offset, tick] {
		download.Invalidate(offset, range.size);
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
		std::lock_guard lock(m_download_mutex);
		const auto done = std::ranges::find_if(m_pending_downloads, [&](const PendingDownload& pending) {
			return pending.address == range.address && pending.size == range.size && pending.tick == tick;
		});
		if (done != m_pending_downloads.end()) m_pending_downloads.erase(done);
	});
	return true;
}

uint64_t TextureCache::PendingDownloadTick(uint64_t address, uint64_t size) {
	std::lock_guard lock(m_download_mutex);
	uint64_t        latest = 0;
	for (const auto& pending: m_pending_downloads) {
		if (address < pending.address + pending.size && pending.address < address + size) {
			latest = std::max(latest, pending.tick);
		}
	}
	return latest;
}

void TextureCache::TouchSpeculated(Image& image, uint64_t serial) {
	std::scoped_lock lock {m_lock};
	if (image.serial != serial || !image.registered) return;
	image.frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	TouchImage(image);
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size, const char* source) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	if (auto* spec = Spec::Current()) {
		spec->NoteInvalidated(address, size); // (when it is committed)
		return;
	}
	std::scoped_lock lock {m_lock};
	const auto images = FindImagesInRegion(address, size, true);
	for (const auto id: images) {
		auto& image = m_slot_images[id];
		if (!image.Overlaps(address, size)) {
			continue;
		}
		if (image.depth_id) {
			auto* depth = m_slot_images.try_get(image.depth_id);
			if (depth != nullptr && depth->info.HasStencil() && depth->backing.image != nullptr) {
				depth->MarkStencilModified();
			}
			continue;
		}
		static const bool log_uploads = std::getenv("KYTY_UPLOAD_LOG") != nullptr;
		if (log_uploads && !image.IsBufferModified()) {
			std::printf("[tsc %llu] BUFFER-WRITE %s write=0x%llx+0x%llx image=0x%llx+0x%llx fmt=%u %ux%u shader=0x%llx\n",
			            static_cast<unsigned long long>(__rdtsc()), source,
			            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
			            static_cast<unsigned long long>(image.info.data.address),
			            static_cast<unsigned long long>(image.info.data.size),
			            static_cast<uint32_t>(image.info.guest_format), image.info.extent.width,
			            image.info.extent.height,
			            static_cast<unsigned long long>(LiveCounters::g_last_dispatch_shader));
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
}

bool TextureCache::HasTrackedDataOverlap(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) return true;
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

bool TextureCache::HasGpuWrittenImageOverlap(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) return true;
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		if (image.IsGpuModified() || image.IsStencilModified()) return true;
	}
	return false;
}

TextureCache::ReadOnlyBufferOverlap TextureCache::ClassifyReadOnlyBufferOverlap(
    uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) return ReadOnlyBufferOverlap::Unsafe;
	std::scoped_lock lock {m_lock};
	const auto images = FindImagesInRegion(address, size, false);
	for (const auto id : images) {
		const auto& image = m_slot_images[id];
		// Only ordinary sampled images whose contents still come from the CPU.
		// In particular, derived depth/stencil views are not covered by the
		// simpler IsRegionGpuModified query and must remain excluded here.
		if (!image.usage.texture || image.usage.storage || image.usage.render_target ||
		    image.usage.depth_target || image.usage.video_out || image.depth_id ||
		    image.info.IsDepth() || image.info.HasStencil() ||
		    image.info.metadata.kind != ImageMetadataKind::None ||
		    image.IsGpuModified() || image.IsStencilModified() || image.IsBufferModified())
			return ReadOnlyBufferOverlap::Unsafe;
	}
	return images.empty() ? ReadOnlyBufferOverlap::None : ReadOnlyBufferOverlap::CpuSampled;
}

bool BufferCache::TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t begin, uint64_t size) {
	if (!GuestRange {begin, size}.Valid()) return false;
	// Follow image -> buffer-region -> page lock order. Hold the image lock
	// through invalidation so registration cannot introduce an alias in between.
	std::unique_lock lock(m_texture_cache.m_lock);
	if (!m_texture_cache.FindImagesInRegion(begin, size, true).empty()) return false;
	// Registration takes the image lock and then this region lock, so handing the image
	// lock over once the region lock is held keeps the check and the state change atomic
	// for it, while the page protection change no longer blocks image lookups.
	const bool handoff = kyty_local_write_window_handoff_mode.load(std::memory_order_relaxed) != 0;
	return m_memory_tracker.TryInvalidateCpuWriteWindow(fault, begin, size, handoff ? &lock : nullptr);
}

bool TextureCache::IsRegionGpuModified(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		// Guest memory holds the CPU's write into the image, and the image can no longer
		// be downloaded: it would only refuse the read. A dead image in reused memory
		// otherwise failed every CPU-side read of the new data until eviction.
		if (!image.depth_id && image.IsGpuModified() && !image.IsDefinitelyCpuDirty()) {
			return true;
		}
	}
	return false;
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = address & ~(TRACKER_PAGE_SIZE - 1);
	const auto page_end   = (address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (owner->Overlaps(address, size)) {
			if (TryInvalidatePartial(*owner, address, size, PartialDirtyGranule)) {
				continue;
			}
			if (SlowLog::Threshold() > 0.0 && owner->info.data.size >= PartialDirtyMinSize &&
			    owner->IsTracked() &&
			    kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) != 0) {
				std::printf("[tsc %llu] FULL-DIRTY addr=0x%llx size=0x%llx write=0x%llx+0x%llx candidate=%d "
				            "(depth_id=%d backing=%d samples=%u volume=%d depth=%d stencil_plane=%d compression=%u) "
				            "registered=%d tracked=%d whole=%d take=%d maybe=%d cpu=%d partial=%d gpu=%d buffer=%d "
				            "stencil=%d\n",
				            static_cast<unsigned long long>(__rdtsc()),
				            static_cast<unsigned long long>(owner->info.data.address),
				            static_cast<unsigned long long>(owner->info.data.size),
				            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
				            PartialDirtyCandidate(*owner) ? 1 : 0, owner->depth_id ? 1 : 0,
			            owner->backing.image != nullptr ? 1 : 0, owner->info.samples,
			            owner->info.IsVolume() ? 1 : 0, owner->info.IsDepth() ? 1 : 0,
			            owner->info.HasStencil() ? 1 : 0,
			            static_cast<uint32_t>(owner->info.metadata.compression), owner->registered ? 1 : 0,
				            owner->IsTracked() ? 1 : 0,
				            owner->track_addr == owner->info.data.address &&
				                    owner->track_addr_end == owner->info.data.End()
				                ? 1
				                : 0,
				            owner->CanTakePartialDirty() ? 1 : 0, owner->IsMaybeCpuDirty() ? 1 : 0,
				            owner->IsDefinitelyCpuDirty() ? 1 : 0, owner->IsPartiallyCpuDirty() ? 1 : 0,
				            owner->IsGpuModified() ? 1 : 0, owner->IsBufferModified() ? 1 : 0,
				            owner->IsStencilModified() ? 1 : 0);
				std::fflush(stdout);
			}
			owner->InvalidateCpuWrite(address, size);
			UntrackImage(id, "cpu-write");
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
	if (!MetaFilterMayHold(address)) return false;
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end() && found->second.type != MetaDataInfo::Type::PendingDcc;
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice, uint32_t* fill_value,
                                 bool* fill_known) {
	// A speculation's answer is as after its holes; one it relies on is noted (speculation-state.h).
	auto* spec = slice < 32 && fill_value == nullptr && fill_known == nullptr ? Spec::Current() : nullptr;
	if (spec != nullptr)
		if (const auto known = spec->KnownMeta(address, slice); known >= 0) return known == 1;
	const auto not_cleared = [&] {
		if (spec != nullptr) spec->NoteMetaRead(address, slice);
		return false;
	};
	if (!MetaFilterMayHold(address)) return not_cleared();
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || found->second.type == MetaDataInfo::Type::PendingDcc ||
	    slice >= 32) {
		return not_cleared();
	}
	if (fill_value != nullptr) {
		*fill_value = found->second.fill_value;
	}
	if (fill_known != nullptr) {
		*fill_known = found->second.fill_known;
	}
	return (found->second.clear_mask & (1u << slice)) != 0 || not_cleared();
}

bool TextureCache::ClearMeta(uint64_t address) {
	if (!MetaFilterMayHold(address)) return false;
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || found->second.type == MetaDataInfo::Type::PendingDcc ||
	    found->second.type == MetaDataInfo::Type::Dcc) {
		// Preserve the broad metadata-clear operation for CMask/FMask/HTile. DCC requires a
		// validated fill value, so an arbitrary compute write must not clear it.
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	found->second.fill_known = false;
	m_meta_epoch.fetch_add(1, std::memory_order_release); // (a clear pending: cached answers ask again)
	return true;
}

bool TextureCache::ClearMeta(uint64_t address, uint32_t fill_value) {
	if (!MetaFilterMayHold(address)) return false;
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || found->second.type == MetaDataInfo::Type::PendingDcc ||
	    found->second.type == MetaDataInfo::Type::Dcc) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	found->second.fill_value = fill_value;
	found->second.fill_known = true;
	m_meta_epoch.fetch_add(1, std::memory_order_release);
	return true;
}

void TextureCache::TrackDccFill(uint64_t address, uint64_t size, uint32_t fill_value) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid DCC fill range\n");
	}
	// DCC fills use a repeated byte code. Require all four bytes of the detected dword to agree,
	// and mark only recognized deferred-clear encodings as logically clear.
	const auto dcc_clear_mask = [fill_value] {
		const auto code = static_cast<uint8_t>(fill_value);
		if (fill_value != static_cast<uint32_t>(code) * 0x01010101u) {
			return 0u;
		}
		switch (code) {
			case 0x00:
			case 0x20:
			case 0x40:
			case 0x80:
			case 0xc0: return UINT32_MAX;
			default: return 0u;
		}
	}();
	std::scoped_lock lock {m_lock};
	// The guest dispatch still writes metadata. An unknown address remains PendingDcc until an
	// image descriptor confirms its role; never reinterpret CMask/FMask/HTile as DCC.
	const auto found = m_surface_metas.try_emplace(address).first;
	MetaFilterAdd(address);
	m_meta_epoch.fetch_add(1, std::memory_order_release);
	if (found->second.type == MetaDataInfo::Type::PendingDcc ||
	    found->second.type == MetaDataInfo::Type::Dcc) {
		found->second.clear_mask = dcc_clear_mask;
		found->second.fill_value = fill_value;
		found->second.fill_size  = size;
		found->second.fill_known = true;
	}
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	if (!MetaFilterMayHold(address)) return false;
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || found->second.type == MetaDataInfo::Type::PendingDcc ||
	    slice >= 32) {
		return false;
	}
	if (is_clear) {
		found->second.clear_mask |= 1u << slice;
		m_meta_epoch.fetch_add(1, std::memory_order_release);
	} else {
		found->second.clear_mask &= ~(1u << slice);
	}
	return true;
}

void TextureCache::MapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid map range\n");
	}
	std::scoped_lock lock {m_lock};
	// Memory mapped under a cached image replaces its contents without a CPU write through the
	// watched pages (the new pages may have been filled through another mapping).
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (TryInvalidatePartial(*owner, address, size, TRACKER_PAGE_SIZE)) {
			continue;
		}
		owner->InvalidateCpuWrite(address, size);
		UntrackImage(id, "map");
	}
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	uint32_t       slow_deleted = 0, slow_partial = 0;
	uint64_t       slow_bytes   = 0, slow_largest = 0, slow_largest_address = 0;
	struct Count {
		uint32_t&                             deleted;
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~Count() {
			LiveCounters::Add(LiveCounters::TextureUnmaps);
			LiveCounters::Add(LiveCounters::TextureUnmapDeletes, deleted);
			LiveCounters::Add(LiveCounters::TextureUnmapUs,
			                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - start)
			                                            .count()));
		}
	} count {slow_deleted};
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW TextureUnmap %.1f ms unmap=0x%llx+0x%llx deleted=%u bytes=0x%llx largest=0x%llx@0x%llx "
		            "partial=%u\n",
		            ms, static_cast<unsigned long long>(address), static_cast<unsigned long long>(size), slow_deleted,
		            static_cast<unsigned long long>(slow_bytes), static_cast<unsigned long long>(slow_largest),
		            static_cast<unsigned long long>(slow_largest_address), slow_partial);
	}, SlowLog::HitchThreshold());
	std::scoped_lock lock {m_lock};
	const auto       end = address + size;
	bool             erased_metadata = false;
	for (auto metadata = m_surface_metas.lower_bound(address);
	     metadata != m_surface_metas.end() && metadata->first < end;) {
		metadata        = m_surface_metas.erase(metadata);
		erased_metadata = true;
	}
	if (erased_metadata) {
		// Each word keeps the bits of the remaining keys (a subset of what it holds).
		std::array<uint64_t, 16> bits {};
		for (const auto& [key, info]: m_surface_metas) {
			const auto bit = MetaFilterBit(key);
			bits[bit >> 6u] |= uint64_t {1} << (bit & 63u);
		}
		for (size_t i = 0; i < bits.size(); ++i) m_meta_filter[i].store(bits[i], std::memory_order_release);
	}
	m_meta_epoch.fetch_add(1, std::memory_order_release);
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		// A texture pool the game streams through keeps its other layers: the game unmaps one
		// layer's memory at a time. The unmapped span becomes dirty (read back as it is when
		// the image is next used) and stays unwatched until then; MapMemory marks it again
		// when memory returns there.
		if ((address > owner->info.data.address || address + size < owner->info.data.End()) &&
		    TryInvalidatePartial(*owner, address, size, TRACKER_PAGE_SIZE)) {
			LiveCounters::Add(LiveCounters::PartialUnmaps);
			++slow_partial;
			continue;
		}
		++slow_deleted;
		slow_bytes += owner->info.data.size;
		if (owner->info.data.size > slow_largest) {
			slow_largest         = owner->info.data.size;
			slow_largest_address = owner->info.data.address;
		}
		if (owner->IsGpuModified()) {
			owner->ClearGpuModified();
		}
		DeleteImage(id);
	}
}

void TextureCache::EraseRetired() {
	std::erase_if(m_retired, [&](const auto& retired) {
		if (!Spec::PacketsPassed(retired.second)) return false;
		m_slot_images.erase(retired.first);
		return true;
	});
}

void TextureCache::RunGarbageCollector() {
	EraseRetired();
	std::scoped_lock lock {m_lock};
	const uint64_t   tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
		if ((tick & 31u) == 0) UpdateGcThresholds();
	}
	// Every 30 seconds: where the video memory stands (the run log).
	static auto last_report = std::chrono::steady_clock::now();
	if (m_graphics.CanReportMemoryUsage() && std::chrono::steady_clock::now() - last_report >= std::chrono::seconds(30)) {
		last_report = std::chrono::steady_clock::now();
		std::printf("Video memory: %llu MiB in use of a %llu MiB budget, %llu MiB of it textures and %llu MiB buffers (collecting from %llu, pressured from %llu, critical from %llu)\n",
		            static_cast<unsigned long long>(m_total_used_memory >> 20u),
		            static_cast<unsigned long long>(m_over_budget_memory >> 20u),
		            static_cast<unsigned long long>(m_cache_bytes >> 20u),
		            static_cast<unsigned long long>(m_buffer_cache.CacheBytes() >> 20u),
		            static_cast<unsigned long long>(m_trigger_gc_memory >> 20u),
		            static_cast<unsigned long long>(m_pressure_gc_memory >> 20u),
		            static_cast<unsigned long long>(m_critical_gc_memory >> 20u));
		std::fflush(stdout);
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	// Past the budget itself (memory already spilling to system RAM): images unused for half a second
	// go, more of them per collection.
	const bool over_budget = m_over_budget_memory != 0 && m_total_used_memory >= m_over_budget_memory;
	if (over_budget) {
		// The freed images kept for reuse (up to 1 GiB) go first: they hold memory nothing draws with.
		m_graphics.TrimImagePool();
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	// dirty_too: images the GPU wrote may go (each one is a synchronous download to guest memory first:
	// under memory pressure these took up to a second a frame). The clean pass runs first; the dirty
	// pass only when the clean pass left the cache still pressured.
	const auto collect = [&](bool allow_aggressive, bool dirty_too) {
		bool           pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool           aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		// Over the budget: images not drawn with in the last 2 frames, oldest first, until the usage is
		// back under the pressure line (else at most a few dozen a frame, which never caught up).
		// (30 frames, not 2: images drawn with a moment ago came straight back as full re-uploads.)
		const uint64_t age        = std::min<uint64_t>(
		    aggressive && over_budget ? 30 : aggressive ? 160 : pressured ? 80 : 16, tick);
		size_t deletions = aggressive && over_budget ? 4096 : aggressive ? 40 : pressured ? 20 : 10;
		std::vector<ImageId> candidates;
		candidates.reserve(std::min<size_t>(deletions, 256));
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
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				continue;
			}
			// The game's streaming texture arrays (320-352 MiB, a layer streamed in at a time) came straight back
			// as full re-uploads when evicted past the budget (79 frames with a 320 MiB upload in a Boletaria run on
			// an 8 GB budget): those go only by the normal 160-frame age.
			// Not even by the 160-frame age: what turning the camera away for three seconds leaves undrawn came back as
			// a 320 MiB upload when it turned back (42 of a run's 43 stalls carried ~330 MiB of full uploads). They go
			// only past the budget, once unused for 20 seconds.
			if (owner->info.data.size >= (128ull << 20) &&
			    (!over_budget || tick - owner->lru_tick < 1200u)) {
				continue;
			}
			if (owner->IsGpuModified()) {
				const bool safe = SafeToDownload(*owner);
				if (safe && owner->info.IsTiled()) {
					continue;
				}
				if (safe && (!pressured || !dirty_too)) {
					continue;
				}
				if (safe && !TryDownloadImage(id)) {
					continue;
				}
				owner->ClearGpuModified();
			}
			if (SlowLog::Threshold() > 0.0 && owner->info.data.size >= PartialDirtyMinSize) {
				std::printf("[tsc %llu] GC-DELETE addr=0x%llx size=0x%llx used=%llu\n",
				            static_cast<unsigned long long>(__rdtsc()),
				            static_cast<unsigned long long>(owner->info.data.address),
				            static_cast<unsigned long long>(owner->info.data.size),
				            static_cast<unsigned long long>(m_total_used_memory));
				std::fflush(stdout);
			}
			DeleteImage(id);
			if (over_budget && m_total_used_memory < m_pressure_gc_memory) break;
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false, false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true, false);
	}
	// Still over the budget after the clean passes: the GPU-written ones too, a few per frame.
	if (over_budget && m_total_used_memory >= m_over_budget_memory) {
		collect(true, true);
	}
}

void TextureCache::ProcessDownloadImages() {
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			(void)TryDownloadImage(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
