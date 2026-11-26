#include <atomic>
#include <memory>
#include <tuple>
#include "local/envio1.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "gpu-residency.h"
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
#include "fresh-memory.h"
#include "guest-lazy-commit.h"
#include "texture-offload.h"
#include "vram-unica.h" // (w6, KYTY_VRAM_UNICA)
#include <atomic> // (H1)
namespace {
std::atomic<uint64_t>& H1TexVmBytes() { // (H1, KYTY_H1_TEX_VM) bytes tomados por formato probado en VirtualMemo
	static std::atomic<uint64_t> b {0};
	return b;
}
} // namespace
#include "buf-nativo.h" // (BUF-NATIVO)
#include "async-upload.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "kernel/pthread.h"
#include "common/hostException.h"
#include "live-counters.h"
#include "local-platform.h"
#include "native-resource-state.h"
#include "scene-tags.h"
#include "graphics/host_gpu/renderer/cache/shadowPoolTrace.h"
#include "loader/demonsSoulsMemMap.h" // (ram1: KYTY_DS_RT_OFFLOAD, the render-target arenas)
#include "common/guardedCopy.h" // (ram15: KYTY_NATIVE_TEXTURE_OFFLOAD, the check's compare)
#include "graphics/host_gpu/hostMemory.h" // (rtq2: HashGuestRange, pages not committed)
#include "ram-native-tex.h" // (ram1b: KYTY_RAM_NATIVE_TEX)
#include "malla-nativa.h" // (MALLA-NATIVA fase 1h: hueco pedido por las mallas)
#include "tex-anticipada.h" // (etapa2: KYTY_TEX_SOLO_GPU)
#include "tex-directa.h" // (etapa 2.1: KYTY_TEX_DIRECTA)
#include "tex-nativa.h" // (N1-N3: KYTY_TEX_NATIVA)
#include "subida-rep.h" // (SUBIDA-REP: KYTY_SUBIDA_REPARTO)
#include "vram-drv.h" // (VRAM-DRV: desglose medido y techo propio del recolector)
#include "vram-juego.h" // (VRAM-JUEGO: KYTY_VRAM_JUEGO, pool de streaming del juego)
#include "vram-tope.h" // (VRAM-TOPE: tope unico por tarjeta, calidad proporcional, cambio de mundo)
#include "tex-stream.h" // (TEX-STREAM: KYTY_TEX_STREAM, mips bajo demanda con el juego)
#include "gestor-memoria.h" // (M7) gestor central de memoria: registro, uso y expulsion por prioridad
#include "graphics/host_gpu/renderer/image/sparseImage.h" // (ram1b)

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <span>
#include <thread>
#include <tuple>
#include <unordered_set> // (RT-NATIVO)
#include <unordered_map> // (VRAM-DESG)
#include <map> // (R2-INTERMEDIAS)
#include <string> // (R2-INTERMEDIAS)
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>
#include "speculation-state.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "soft-release.h"
#include "ram-guard.h"

[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_write_window_handoff_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_image_granules_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_texture_resolve_pages_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_partial_image_dirty_mode {0};
// 1: a partial refresh uploads only the rows of tile blocks over the dirty ranges of a large
// subresource, not the whole subresource (UploadImagePartial).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_partial_row_bands_mode {0};
// 0: a sampled view refreshes the whole texture, not only its levels (RefreshTextureLevels; on by default, 0 for an A/B).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_texture_levels_mode {1};
// 0: an image GPU buffer writes covered parts of is uploaded whole, not only the subresources over them (InitializeImage;
// on by default, 0 for an A/B).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_buffer_ranges_mode {1};
// 0: a compute fill over images it does not match runs as the dispatch, and the images are uploaded after it, not cleared
// (TryConsumeComputeImageClear, InitializeImage; on by default, 0 for an A/B; KYTY_FILL_CLEAR=0 from the start).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_fill_clear_mode {[] {
	const char* value = std::getenv("KYTY_FILL_CLEAR");
	return value != nullptr && value[0] == '0' ? 0u : 1u;
}()};

namespace Libs::Graphics {

void VramDrvReport(const GraphicContext& g); // vma.cpp (VRAM-DRV)
uint64_t VramCardBytes(); // vma.cpp (VRAM-TOPE)

void DrainImageDestroys(); // (DESTRUIR-HILO, vma.cpp)

uint64_t BufferCacheStagingRingSize(); // bufferCache.cpp

namespace {

// Guest frames an overlapping image must stay unused before a lookup of another image may delete
// it. (Compared with scheduler ticks before, dozens a frame: a 4K target and a 2560x1440 one the
// game renders into the same transient memory every frame deleted and re-created each other,
// 64 MiB of page tracking both ways, ~54 times a second.)
constexpr uint64_t NumFramesBeforeRemoval = 32;

// KYTY_PARTIAL_IMAGE_DIRTY: a game streaming textures into a large array writes one layer at a
// time. The whole-image path stopped watching all of the image's pages on the first write and
// then re-watched, copied, detiled and uploaded all of it (a 352 MiB array: 20-70 ms on the
// render thread, several times a second while walking into a new area). Such images keep
// watching everything but the written granules and upload only the subresources over them.
// (Size from which the diagnostic logs report images.)
constexpr uint64_t PartialDirtyMinSize = uint64_t {16} << 20;
// A write fault releases the image's granule around it (granules start at the image start):
// a streamed layer faults a few times instead of once per 4 KiB page.
constexpr uint64_t PartialDirtyGranule = uint64_t {1} << 20;
// Every image of two granules or more takes partial writes: below that one granule is the whole image. Mip-streamed
// textures (5-11 MiB BC: the game writes the levels while draws already sample it) uploaded all of it at each use
// between the writes (2-10 times in a few ms when entering an area).
constexpr uint64_t PartialDirtyCandidateSize = 2 * PartialDirtyGranule;
// A refresh whose subresources cover more than this share of the image uploads all of it.
constexpr uint64_t PartialUploadMaxShare = 2;
// Staged runs of picked subresources closer than this are staged as one run.
constexpr uint64_t PartialStageGap = uint64_t {64} << 10;
// UploadImageChunked: images from this size whose bytes no cached buffer holds whole, uploaded this much of guest
// memory at a time.
constexpr uint64_t ChunkedUploadMin   = uint64_t {256} << 20;
constexpr uint64_t ChunkedUploadPiece = uint64_t {64} << 20;

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

namespace {
constexpr uint32_t ImageGranuleBits  = 16;
constexpr uint64_t ImageGranuleSpace = uint64_t {1} << 40; // the image page table's address space
} // namespace

static TextureCache* g_report_cache = nullptr;

void TextureCache::WriteReport(const char* path) {
	FILE* file = std::fopen(path, "wb");
	if (file == nullptr) return;
	std::fputs("id\taddress\tguest_size\tbytes\tformat\tguest_format\ttype\twidth\theight\tdepth\tlevels\tlayers\tsamples\ttile\tusage\tregistered\tgpu_modified\tlru_tick\tgc_tick\tmemory_type\n", file);
	std::scoped_lock lock {m_lock};
	m_slot_images.ForEach([&](ImageId id, Image& image) {
		VmaAllocationInfo allocation {};
		if (image.backing.allocation != nullptr) vmaGetAllocationInfo(m_graphics.allocator, image.backing.allocation, &allocation);
		const auto& usage = image.usage;
		char        kinds[8] {};
		size_t      k = 0;
		if (usage.texture) kinds[k++] = 't';
		if (usage.storage) kinds[k++] = 's';
		if (usage.render_target) kinds[k++] = 'r';
		if (usage.depth_target) kinds[k++] = 'd';
		if (usage.video_out) kinds[k++] = 'v';
		if (k == 0) kinds[k++] = '-';
		std::fprintf(file, "%u\t0x%" PRIx64 "\t%" PRIu64 "\t%" PRIu64 "\t%s\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%s\t%d\t%d\t%" PRIu64 "\t%" PRIu64 "\t%d\n",
		             id.index, image.info.data.address, image.info.data.size, static_cast<uint64_t>(allocation.size),
		             vk::to_string(image.backing.format).c_str(), static_cast<uint32_t>(image.info.guest_format),
		             static_cast<uint32_t>(image.info.type), image.info.extent.width, image.info.extent.height,
		             image.info.extent.depth, image.info.resources.levels, image.info.resources.layers, image.info.samples,
		             static_cast<uint32_t>(image.info.tile_mode), kinds, image.registered ? 1 : 0, image.IsGpuModified() ? 1 : 0,
		             image.registered ? m_lru_cache.TickOf(image.lru_id) : uint64_t {0}, m_gc_tick,
		             image.backing.allocation != nullptr ? static_cast<int>(allocation.memoryType) : -1);
	});
	std::fclose(file);
}

void TextureCache::CollectResidencyRanges(std::vector<GpuResidency::Range>& out, uint64_t& count) {
	count = 0;
	std::scoped_lock lock {m_lock};
	m_slot_images.ForEach([&](ImageId /*id*/, const Image& image) {
		if (!image.registered || image.info.data.address == 0 || image.info.data.size == 0) return;
		const auto& usage = image.usage;
		const auto  kind  = usage.render_target || usage.depth_target || usage.video_out ? GpuResidency::Target
		                    : usage.storage || image.IsGpuModified()                    ? GpuResidency::Storage
		                                                                                : GpuResidency::Texture;
		out.push_back({image.info.data.address, image.info.data.size, kind});
		count++;
	});
}

// KYTY_TEXTURE_OFFLOAD (texture-offload.h): the cache its restores go to, while it lives.
static TextureCache*     g_offload_cache = nullptr;
static std::atomic<bool> g_offload_alive {false};

// A check of images' downloads against their guest bytes (OffloadScan): downloaded into one buffer, compared by the
// offload worker when the GPU is done.
struct TextureCache::OffloadCheck {
	struct Entry {
		uint64_t    serial = 0, address = 0, begin = 0, end = 0, offset = 0; // (offset: of the image's first byte, `address`)
		uint64_t    kind = 0;                                                // (OffloadKind)
		std::string what;                                                    // (for the log)
		uint64_t    format = 0;                                              // (OffloadFormat)
		// (ram3, KYTY_OFFLOAD_PARTIAL) The mapped runs [first, second) compared when not all of [begin, end) is mapped.
		std::vector<std::pair<uint64_t, uint64_t>> runs;
	};
	struct Result {
		uint64_t serial = 0, kind = 0;
		bool     equal = false, known = true; // (known: its guest bytes were there to compare)
		uint64_t format = 0;                  // (OffloadFormat)
		// (w6, KYTY_VRAM_UNICA) With some pieces different: the pieces equal to the download (taken alone).
		std::vector<std::pair<uint64_t, uint64_t>> equal_runs;
		// (w12) And the different ones: uploaded again from guest memory, then checked once more.
		std::vector<std::pair<uint64_t, uint64_t>> differ_runs;
	};
	std::unique_ptr<Buffer> buffer, scratch; // (system memory; video memory the images are detiled into first)
	std::vector<Entry>      entries;
	std::vector<Result>     results;
	std::atomic<bool>       done {false};
};

// An asynchronous restore (RestoreOffloaded): the images' bytes downloaded into one buffer, written back to the ranges by
// the offload worker when the GPU is done.
struct TextureCache::OffloadRestore {
	struct Copy {
		size_t   range  = 0;
		uint64_t source = 0; // (offset of the range's first byte in the buffer)
	};
	std::unique_ptr<Buffer>                buffer, scratch; // (as OffloadCheck's)
	std::vector<TextureOffload::Restoring> ranges;
	std::vector<Copy>                      copies;
	uint64_t                               bytes = 0, lost = 0;
	std::chrono::steady_clock::time_point  start;
	std::atomic<bool>                      done {false};
};

// (ram1b, KYTY_RAM_NATIVE_TEX=2) A sparse image's bound layers against a dense copy of the same description uploaded
// from guest memory by the current path: both downloaded, every level of each layer, into one buffer (the reference's
// first, then the sparse image's at `half`), compared by layer (XXH3) by the offload worker when the GPU is done.
struct TextureCache::RntCheck {
	std::unique_ptr<Buffer>                   buffer;
	uint64_t                                  half = 0;
	std::vector<std::tuple<uint32_t, uint64_t, uint64_t>> layers; // (layer, offset, bytes)
	bool                                      proven = false;     // its format proven exact by an offload check
	std::string                               what;
	std::atomic<bool>                         done {false};
};

// The checks' compares, the asynchronous restores' write-backs and the discards of taken ranges: on a thread of their
// own (the first two handed over by the priority thread when the GPU is done, where they would hold up the game's
// end-of-pipe and flip operations; the discards took 10-90 ms of the render thread a scan).
class OffloadWorker {
public:
	static void Post(std::function<void()> job) {
		static auto* worker = new OffloadWorker; // (never destroyed: its thread may outlive static destructors)
		{
			std::lock_guard lock(worker->m_mutex);
			worker->m_jobs.push_back(std::move(job));
		}
		worker->m_ready.notify_one();
	}

private:
	OffloadWorker() {
		std::thread([this] {
			LocalPlatform::SetThreadName("Kyty.Offload");
			std::vector<std::function<void()>> jobs;
			for (;;) {
				{
					std::unique_lock lock(m_mutex);
					m_ready.wait(lock, [this] { return !m_jobs.empty(); });
					jobs.swap(m_jobs);
				}
				for (auto& job: jobs) job();
				jobs.clear();
			}
		}).detach();
	}
	std::mutex                         m_mutex;
	std::condition_variable            m_ready;
	std::vector<std::function<void()>> m_jobs;
};

// What makes two images' downloads the same transformation of their bytes: format, tiling, extent, levels, layers.
[[maybe_unused]] static uint64_t OffloadKind(const Image& image) {
	const auto& info = image.info;
	uint64_t    kind = static_cast<uint64_t>(info.guest_format);
	for (const uint64_t value: {static_cast<uint64_t>(info.tile_mode), static_cast<uint64_t>(info.extent.width),
	                            static_cast<uint64_t>(info.extent.height), static_cast<uint64_t>(info.extent.depth),
	                            static_cast<uint64_t>(info.resources.levels), static_cast<uint64_t>(info.resources.layers),
	                            static_cast<uint64_t>(info.bgra16 ? 1 : 0)})
		kind = kind * 0x100000001b3ull ^ value;
	return kind;
}

// KYTY_GPU_ONLY_OFFLOAD: what makes two images' downloads the same transformation of their bytes but for their size
// (format, tiling): a texture check that gave a format's guest bytes back exactly proves a GPU-written image's restore.
[[maybe_unused]] static uint64_t OffloadFormat(const Image& image) {
	const auto& info = image.info;
	return (static_cast<uint64_t>(info.guest_format) * 0x100000001b3ull ^ static_cast<uint64_t>(info.tile_mode)) * 0x100000001b3ull ^
	       static_cast<uint64_t>(info.bgra16 ? 1 : 0);
}

// KYTY_GPU_ONLY_OFFLOAD (texture-offload.h): with the texture offload on, the guest bytes of GPU-written images taken too
// (=1; default: with the low-RAM profile, or while the RAM guard's pressure is critical; =0 never).
[[maybe_unused]] static bool GpuOnlyOffloadActive() {
	if (TextureOffload::GetMode() != TextureOffload::Mode::Offload) return false;
	const int setting = TextureOffload::GpuOnlySetting();
	if (setting >= 0) return setting != 0;
	return RamGuard::LowRam() || RamGuard::PressureLevel() >= RamGuard::Critical;
}

// (ram1) KYTY_DS_RT_OFFLOAD=1|2 (Demon's Souls, off by default; needs KYTY_GPU_ONLY_OFFLOAD active): a GPU-written image
// wholly inside the game's render-target arenas (1: RTSharedAllocator, the 687 MiB the frame's targets alias in; 2: and
// RTPersistAllocator, the 897 MiB of targets kept across frames), found by the memory map (KYTY_MEMMAP), needs no
// texture check of its format to have its guest bytes dropped (the "format unproven" bytes of "Guest RAM by GPU
// cache"). Ghidra 01.007.000: the engine reserves both arenas whole at start-up for GPU work only (PS5CRenderDisplay
// 0x90ad80, tables 0x24f3e80 / 0x24f8c30) and its CPU code never reads a target's bytes [H: but for a capture or
// thumbnail]; a native port keeps no CPU copy of them. Every other GPU-only gate stays (not depth, dirty, linear,
// pending a read back or under GPU buffer writes; a download plan). A touch still restores them by a download (the bytes
// a CPU read back would get), only not proven exact for that format.
[[maybe_unused]] static int DsRtOffloadMode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_DS_RT_OFFLOAD");
		const int   m = v != nullptr && (v[0] == '1' || v[0] == '2') ? v[0] - '0' : 0;
		if (m != 0) {
			std::printf("DS RT offload: on (KYTY_DS_RT_OFFLOAD=%d): GPU-written images inside RTShared%s go out of RAM with the "
			            "GPU-only offload without a format check (needs KYTY_MEMMAP and the GPU-only offload)\n",
			            m, m == 2 ? " and RTPersist" : "");
			std::fflush(stdout);
		}
		return m;
	}();
	return mode;
}
// (RT-ALIAS-MEM) KYTY_RT_ALIAS_MEM=1 (por defecto 0 hasta validarlo): un render target / profundidad / storage nuevo cuyo
// rango del juego cae entero dentro del de otra imagen viva de la misma arena RT (RTSharedAllocator o RTPersistAllocator)
// se liga a la VkDeviceMemory de esa otra (aliasing de memoria de Vulkan, image.cpp TryJoinMemAlias) en vez de pedir la
// suya. Ghidra 01.007.000: 0x91c680 (RenderTargetManager::InitSharedMemory) crea las dos arenas como
// RenderTargetArenaAllocator (vtable 0x25afdf0; RTShared con la marca de compartida +0x98 = 1, RTPersist 0); su Reset
// (slot 4, 0x915930) vuelve el cursor al principio: asignacion lineal que el juego rehace, y los RT de distintas vistas y
// pases caen en los mismos bytes (ViewSharedRenderTargets, 0x91a8f0). En PS5 esos RT comparten memoria; el emulador les
// daba una asignacion a cada uno (log 20261125-195442: RTShared 385 MiB de imagenes sobre 238 MiB del juego, 147
// solapados; RTPersist 306 sobre 235, 71 solapados). Linea [rt-alias-mem] cada 30 s con lo ahorrado.
static bool RtAliasMemOn() {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_RT_ALIAS_MEM");
		const bool  r  = v != nullptr && *v == '1';
		std::printf("RT alias de memoria: %s (KYTY_RT_ALIAS_MEM=%s): RT de las arenas del juego contenidos en otro vivo "
		            "comparten su VkDeviceMemory (aliasing de Vulkan, layout UNDEFINED tras pisarse)\n",
		            r ? "on" : "off", v != nullptr ? v : "-");
		std::fflush(stdout);
		return r;
	}();
	return on;
}

[[maybe_unused]] static bool DsRtArena(const Image& image) {
	const int mode = DsRtOffloadMode();
	if (mode == 0 || image.info.data.Empty()) return false;
	const int arena = Loader::DemonsSoulsMemMap::RtArenaOf(image.info.data.address, image.info.data.End());
	return arena == 1 || (arena == 2 && mode == 2);
}

// The images a GPU write made the owner of their bytes (their guest copy is stale or the same).
[[maybe_unused]] static bool GpuWrittenImage(const Image& image) {
	return image.IsGpuModified() || image.usage.render_target || image.usage.storage || image.usage.depth_target;
}

// KYTY_NATIVE_MEMORY=1 (PageManager::NativeWriteWatch): the images over pages of the range the write watch saw written
// while write-watched are invalidated as their write fault invalidated them (GpuResourceManager::HandleFault). Before
// every image lookup, outside m_lock (InvalidateMemory takes it). Nothing otherwise.
static void TakeImageCpuWrites(PageManager& page_manager, TextureCache& cache, uint64_t address, uint64_t size) {
	if (!PageManager::NativeWriteWatch() || size == 0 || !GuestRange {address, size}.Valid()) return;
	thread_local std::vector<PageManager::DeferredRange> runs;
	runs.clear();
	page_manager.TakeCpuWrites(address, size, PageManager::WriteConsumer::Images, &runs);
	for (const auto& run: runs) cache.InvalidateMemory(run.address, run.size);
}

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()) {
	(void)BlockClassOverlapReplaceOn(); // (its start-up line)
	(void)SampleHtileClearOn();         // (its start-up line)
	m_image_granules.assign((ImageGranuleSpace >> ImageGranuleBits) / 64, 0);
	g_report_cache               = this;
	LiveCounters::g_image_report = [](const char* path) {
		if (g_report_cache != nullptr) g_report_cache->WriteReport(path);
	};
	// Collection starts where the video memory use reaches the budget less its headroom (GetTotalMemoryBudget):
	// below it a kept image costs nothing, a deleted one an upload when the game uses it again.
	m_trigger_gc_memory       = m_graphics.GetTotalMemoryBudget();
	m_start_trigger_gc_memory = m_trigger_gc_memory;
#if defined(_WIN32)
	if (TextureOffload::Enabled()) {
		const bool offload = TextureOffload::GetMode() == TextureOffload::Mode::Offload;
		g_offload_cache    = this;
		g_offload_alive.store(true, std::memory_order_release);
		auto& state = TextureOffload::Get();
		state.remove.store(+[](uint64_t vaddr, uint64_t size) { g_offload_cache->m_page_manager.UpdateReadWatchers<false>(vaddr, size); });
		if (offload) {
			// A restore on the render thread; from a game thread through it (as a guest readback: BufferCache::ReadMemory).
			state.restore.store(+[](uint64_t vaddr, uint64_t size, TextureOffload::Touch touch) {
				auto* cache = g_offload_cache;
				if (cache == nullptr || !g_offload_alive.load(std::memory_order_acquire)) return;
				if (GuestGpu::IsGpuThread()) {
					cache->RestoreOffloaded(vaddr, size, static_cast<uint8_t>(touch));
					return;
				}
				cache->m_scheduler.Context().GetGpu().SendCommandSync(
				    [cache, vaddr, size, touch] { cache->RestoreOffloaded(vaddr, size, static_cast<uint8_t>(touch)); });
			});
			// Not: the upload worker (the render thread waits for it), deferred GPU work, a speculation's recorder (they
			// cannot wait for the GPU), other host threads (the render thread may wait for them).
			state.can_wait.store(+[] {
				if (AsyncUpload::t_worker || CommandScheduler::InDeferredOperation() || CommandScheduler::ThreadRecorder() != nullptr)
					return false;
				return GuestGpu::IsGpuThread() || LibKernel::PthreadSelfOrNull() != nullptr;
			});
			// (ram15) KYTY_NATIVE_TEXTURE_OFFLOAD: the guest pages themselves dropped, written back and opened by memory.cpp;
			// every window of the emulator's over guest memory asks for dropped bytes first.
			if (TextureOffload::Native()) {
				state.native_drop.store(&LibKernel::Memory::NativeOffloadDrop, std::memory_order_release);
				state.native_write.store(&LibKernel::Memory::NativeOffloadWrite, std::memory_order_release);
				state.native_open.store(&LibKernel::Memory::NativeOffloadOpen, std::memory_order_release);
				// (ram17, KYTY_RAM17: the disk store reads the pages through a window and forgets its own write-backs.)
				state.native_read.store(&LibKernel::Memory::NativeGuardedRead, std::memory_order_release);
				state.native_forget.store(&LibKernel::Memory::NativeForgetWrites, std::memory_order_release);
				LibKernel::Memory::SetNativeWindowHook(&TextureOffload::NativeBeforeWindow);
			}
		}
		Common::HostException::SetProbeHandler(TextureOffload::OnFault);
		std::printf(offload ? "Texture offload: on (the guest bytes of clean sampled textures of 256 KiB or more dropped from RAM "
		                      "after 2 s clean and a download that gives them back exactly; KYTY_TEXTURE_OFFLOAD=0 keeps them)\n"
		                    : "Texture offload probe: on (clean sampled textures of 256 KiB or more made inaccessible after 2 s "
		                      "clean; nothing discarded)\n");
	}
#endif
}

void KytyTextureGcThresholds(uint64_t total_budget, uint64_t& trigger, uint64_t& pressure, uint64_t& critical) {
	constexpr int64_t GiB       = 1024ll * 1024 * 1024;
	const auto        budget    = static_cast<int64_t>(std::min<uint64_t>(total_budget, INT64_MAX));
	const auto        threshold = std::min<int64_t>(budget, 8 * GiB);
	pressure =
	    static_cast<uint64_t>(std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
	critical =
	    static_cast<uint64_t>(std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
	trigger = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
}

bool KytyGcLiveBudget() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_GC_LIVE_BUDGET");
		return env != nullptr && std::atoi(env) != 0;
	}();
	return enabled;
}

uint64_t KytyGcLiveBudgetValue(const GraphicContext& graphics) {
	static const uint64_t cap = [] {
		const char* env = std::getenv("KYTY_GC_BUDGET_CAP_MB");
		return env != nullptr ? std::strtoull(env, nullptr, 10) << 20 : 0ull;
	}();
	const uint64_t budget     = graphics.GetTotalMemoryBudget();
	const uint64_t now_budget = cap != 0 ? std::min(budget, cap) : budget;
	static std::mutex                        mutex;
	static std::array<uint64_t, 15>          samples {};
	static size_t                            next = 0;
	static std::chrono::steady_clock::time_point last {};
	std::scoped_lock                        lock {mutex};
	const auto                              now = std::chrono::steady_clock::now();
	if (samples[0] == 0) {
		samples.fill(now_budget);
		last = now;
	} else if (now - last >= std::chrono::seconds(2)) {
		next          = (next + 1) % samples.size();
		samples[next] = now_budget;
		last          = now;
	} else {
		samples[next] = std::min(samples[next], now_budget);
	}
	return *std::min_element(samples.begin(), samples.end());
}

TextureCache::~TextureCache() {
	g_offload_alive.store(false, std::memory_order_release); // (no restores: the images go)
	// (Jobs of the priority thread and the offload worker may still name them: never freed.)
	for (auto& check: m_offload_checks) (void)check.release();
	for (auto& restore: m_offload_restores) (void)restore.release();
	LiveCounters::g_image_report = nullptr;
	g_report_cache               = nullptr;
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

// (GC-GPU-SUCIA) KYTY_GC_GPU_SUCIA (por defecto encendido; =0 apagado): una imagen que la GPU pinto (IsGpuModified) y que
// tiene paginas escritas por la CPU (IsCpuDirty, sin escritura de buffer de la GPU encima) no se puede descargar
// (SafeToDownload falso); expulsarla borraba lo pintado por la GPU sin bajarlo y al volver se rehacia desde la RAM del
// juego con datos viejos ("cosas claras" con la VRAM al limite, logs 20261125-033425 y 035522: 2663-4879 expulsiones).
// Encendido: esas imagenes no se expulsan (como las tiled descargables). Linea [gc-gpu-sucia] cada 30 s: las borradas
// sin descarga por motivo (CPU / buffer de la GPU, este ultimo correcto: los datos nuevos estan en el buffer) y las
// retenidas.
static bool GcGpuSucia() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_GC_GPU_SUCIA");
		return v != nullptr && v[0] == '1'; // (04:20) apagado por defecto: retenia imagenes y la VRAM pasaba el presupuesto (log 041308)
	}();
	return on;
}
static std::atomic<uint64_t> g_gcs_cpu {0}, g_gcs_cpu_bytes {0}, g_gcs_buf {0}, g_gcs_buf_bytes {0}, g_gcs_ret {0}, g_gcs_ret_bytes {0};
// Una imagen pintada por la GPU que el recolector debe dejar (encendido): pintada, sucia por la CPU, sin buffer encima.
static bool GcGpuSuciaRetiene(const Image& image) {
	if (!GcGpuSucia() || !image.IsGpuModified() || image.IsBufferModified() || !image.IsCpuDirty()) return false;
	g_gcs_ret.fetch_add(1, std::memory_order_relaxed);
	g_gcs_ret_bytes.fetch_add(image.info.data.size, std::memory_order_relaxed);
	return true;
}
// Una imagen pintada por la GPU borrada sin descarga: contada por su motivo.
static void GcGpuSuciaPerdida(const Image& image) {
	if (image.IsBufferModified()) {
		g_gcs_buf.fetch_add(1, std::memory_order_relaxed);
		g_gcs_buf_bytes.fetch_add(image.info.data.size, std::memory_order_relaxed);
	} else {
		g_gcs_cpu.fetch_add(1, std::memory_order_relaxed);
		g_gcs_cpu_bytes.fetch_add(image.info.data.size, std::memory_order_relaxed);
	}
}

// (N3-REAL, 11-25) Con el tope (KYTY_VRAM_TOPE, por defecto) el recolector y ReclaimImageMemory (sin `all`) NUNCA expulsan
// un render target, una profundidad (mapa de sombras D32 y cascadas incluidos), una imagen de almacenamiento ni nada que
// la GPU haya pintado. Log 20261125-092605: [gc-gpu-sucia] "pintadas por la GPU borradas sin descarga: por escritura de la
// CPU 70 (422 MiB)": al volver se rehacian desde la RAM del juego con datos viejos = sombras/luz "claras". Lo unico
// expulsable es lo que se puede rehacer igual: texturas subidas desde la RAM del juego.
static bool GcProtegida(const Image& image) {
	return image.IsGpuModified() || image.usage.render_target || image.usage.depth_target || image.usage.storage ||
	       image.usage.video_out || image.info.IsDepth();
}

// (RT-GRANDE-VIEJO, 11-25) Log 20261125-105844: [vram-desg] RT de 3840x2160 f97 (63 MiB) en arena 0 (paginas GPU del
// juego, fuera de RTShared/RTPersist) con edad 543-1108 fotogramas y la VRAM 865 MiB sobre el presupuesto; GcProtegida
// los dejaba para siempre. Solo un RT de color (no profundidad, no imagen de flip/VideoOut, no nativa/dispersa) de mas de
// 1920x1080 y >= 4 MiB, fuera de las arenas (en ellas la memoria la comparten otras imagenes), sin usar >= RtGrandeTicks
// (~10 s a 60 fps). Si la GPU lo pinto, el recolector lo descarga antes (TryDownloadImage); si no se puede descargar,
// se queda. KYTY_RT_GRANDE_VIEJO=0: apagado.
constexpr uint64_t RtGrandeTicks = 600;
// (W14) Vistas de formato en vez de imagenes duplicadas (ver FindImage). Linea [vista-formato] cada 30 s: imagenes
// distintas servidas como vista y MiB que no se duplicaron.
namespace VistaFormato {
static bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VISTA_FORMATO");
		const bool  r = v == nullptr || *v != '0';
		std::printf("[vista-formato] %s (KYTY_VISTA_FORMATO=0: apagado)\n", r ? "encendido" : "apagado");
		std::fflush(stdout);
		return r;
	}();
	return on;
}
static std::mutex                   mutex;
static std::unordered_set<uint64_t> vistas; // direccion de la imagen ya contada
static uint64_t                     bytes = 0;
static uint64_t                     usos  = 0;
static std::chrono::steady_clock::time_point ultimo {};
static void Cuenta(const Image& image) {
	std::lock_guard lock(mutex);
	usos++;
	if (vistas.insert(image.info.data.address).second) bytes += image.info.data.size;
	const auto now = std::chrono::steady_clock::now();
	if (now - ultimo >= std::chrono::seconds(30)) {
		ultimo = now;
		std::printf("[vista-formato] 30 s: imagenes con otro formato servidas como vista %zu (%llu MiB sin copia), usos %llu\n",
		            vistas.size(), static_cast<unsigned long long>(bytes >> 20u), static_cast<unsigned long long>(usos));
		std::fflush(stdout);
	}
}
} // namespace VistaFormato
// (W14) Causa de que no saliera ninguna [rt-grande-viejo] (log 20261125-120123): los dos RT de 63 MiB estan pintados por la
// GPU y el bufer de descarga es de 32 MiB (download_max = 16 MiB): el recolector los dejaba como protegidos. Ahora se
// descargan con un bufer propio del tamano exacto (solo para estos RT, hasta 256 MiB), se escriben en la RAM del juego
// (unica copia) y la imagen se borra; si el juego lo vuelve a usar, se resube desde esos bytes.
constexpr uint64_t RtGrandeDescargaMax = uint64_t {256} << 20u;
static bool g_rtg_descarga_propia = false;
static bool RtGrandeViejo(const Image& image, uint64_t used_tick, uint64_t tick) {
	static const bool on = [] { const char* v = std::getenv("KYTY_RT_GRANDE_VIEJO"); return v == nullptr || *v != '0'; }();
	if (!on || image.info.data.Empty() || image.IsSparse() || image.depth_id) return false;
	const auto& u = image.usage;
	if (!u.render_target || u.depth_target || u.video_out || u.storage || image.info.IsDepth()) return false;
	// (W14) Nunca un mapa de sombras: los del juego son cuadrados (2048x2048, cascadas); solo RT de pantalla.
	if (image.info.extent.width == image.info.extent.height) return false;
	if (image.info.extent.width <= 1920 && image.info.extent.height <= 1080) return false;
	if (image.info.data.size < (uint64_t {4} << 20u) || used_tick + RtGrandeTicks > tick) return false;
	if (TexNativa::AnyOwned(image.info.data.address, image.info.data.End())) return false;
	return Loader::DemonsSoulsMemMap::RtArenaOf(image.info.data.address, image.info.data.End()) == 0;
}

// Collector runs (flips) before memory an image could not be made for is asked for again.
constexpr uint64_t FailedRetryTicks = 30;

// KYTY_TEST_RECLAIM=<n>: the first n images that must exist, of 1 MiB and more, asked for from the 600th flip on, are
// taken as not made once (ReclaimImageMemory, then made again); KYTY_TEST_RECLAIM_ROUNDS=2: twice (the second round's
// downloads too).
[[gnu::noinline]] static bool TestReclaim(const ImageInfo& info, uint64_t tick) {
	static std::atomic<int64_t> left {[] {
		const char* value = std::getenv("KYTY_TEST_RECLAIM");
		return value != nullptr ? std::strtoll(value, nullptr, 10) : int64_t {0};
	}()};
	return tick >= 600 && info.data.size >= (uint64_t {1} << 20u) && left.load(std::memory_order_relaxed) > 0 &&
	       left.fetch_sub(1, std::memory_order_relaxed) > 0;
}

static uint32_t TestReclaimRounds() {
	static const uint32_t rounds = [] {
		const char* value = std::getenv("KYTY_TEST_RECLAIM_ROUNDS");
		return value != nullptr ? std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1, 2) : 1u;
	}();
	return rounds;
}

// (VRAM-REPARTO) Lo expulsado por el recolector (direccion -> bytes) para contar lo que vuelve a subirse: la linea
// [vram-reparto] cada 30 s (expulsiones/s, MB re-subidos/s, presupuesto efectivo).
namespace VramReparto {
static std::mutex                             mutex;
static std::unordered_map<uint64_t, uint64_t> evicted;
static std::atomic<uint64_t>                  reup_n {0};
static std::atomic<uint64_t>                  reup_bytes {0};
static std::unordered_map<uint64_t, uint64_t> reup_recent; // (VRAM-DRV2) direccion -> ms en que se volvio a subir
static uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
// (VRAM-DRV2) Vuelta a subir tras expulsarse hace menos de VramDrv::ReupProtegeMs: no se expulsa otra vez.
static bool ReupReciente(uint64_t address, uint64_t now_ms) {
	std::scoped_lock lock {mutex};
	if (reup_recent.empty()) return false;
	const auto it = reup_recent.find(address);
	if (it == reup_recent.end()) return false;
	if (now_ms - it->second < VramDrv::ReupProtegeMs) return true;
	reup_recent.erase(it);
	return false;
}
static void Evicted(uint64_t address, uint64_t size) {
	std::scoped_lock lock {mutex};
	if (evicted.size() >= 65536) evicted.clear();
	evicted[address] = size;
}
static void Created(uint64_t address) {
	std::scoped_lock lock {mutex};
	if (evicted.empty()) return;
	if (auto it = evicted.find(address); it != evicted.end()) {
		reup_n.fetch_add(1, std::memory_order_relaxed);
		reup_bytes.fetch_add(it->second, std::memory_order_relaxed);
		VramDrv::g_reup_n.fetch_add(1, std::memory_order_relaxed);
		VramDrv::g_reup_bytes.fetch_add(it->second, std::memory_order_relaxed);
		GestorMem::Resubido(GestorMem::Vram, it->second); // (M7) lo expulsado volvio: el gestor sube la edad minima
		if (reup_recent.size() >= 65536) reup_recent.clear();
		reup_recent[address] = NowMs();
		evicted.erase(it);
	}
}
} // namespace VramReparto

namespace {
uint64_t HashGuestRange(uint64_t vaddr, uint64_t size); // (TEX-STREAM) definida mas abajo
} // namespace

// (M7, TEX-DEDUP) KYTY_TEX_DEDUP=1 (por defecto; =0 apagado). Deduplicacion real: indice de contenido de texturas
// muestreadas (>= 1 MiB, de color, una capa, sin metadatos, no nativas): mismo hash del contenido, formato y forma en
// OTRA direccion = la imagen nueva se crea como alias de la VkImage de la que ya esta (Image::SetShareSource; cuenta de
// referencias en Image::Share). Cada direccion vigila sus paginas (su propia imagen, TrackImage); si el juego escribe en
// una (o la GPU, o se usa como RT/storage), FindImage borra ese miembro antes de usarlo y se crea otra vez con su propia
// VkImage (copy-on-write). Indice bajo m_lock. Linea [tex-dedup] cada 5 s (GestorPass).
namespace TexDedup {
static bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_DEDUP");
		const bool  r = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("[tex-dedup] KYTY_TEX_DEDUP=%d: texturas con el mismo contenido en otra direccion comparten la VkImage "
		            "(copy-on-write al escribirse)\n",
		            r ? 1 : 0);
		std::fflush(stdout);
		return r;
	}();
	return on;
}
struct Entrada {
	ImageId  id {};
	uint64_t serial = 0;
};
static std::unordered_map<uint64_t, Entrada> indice; // clave contenido+forma -> imagen con ese contenido (bajo m_lock)
static bool Apta(const ImageInfo& info) {
	return info.data.size >= (uint64_t {1} << 20u) && !info.IsDepth() && !info.HasMetadata() && !info.HasStencil() &&
	       info.samples == 1 && info.resources.layers == 1 && !info.IsVolume() &&
	       info.metadata.compression == VideoOutCompression::Uncompressed &&
	       !(TexNativa::On() && TexNativa::IsNative(info.data.address));
}
static uint64_t Clave(const ImageInfo& info, uint64_t hash) {
	return hash ^ (uint64_t {static_cast<uint32_t>(info.pixel_format)} << 40u) ^ (uint64_t {info.extent.width} << 20u) ^
	       info.extent.height ^ (uint64_t {info.resources.levels} << 56u) ^ info.data.size * 0x9E3779B97F4A7C15ull ^
	       (uint64_t {static_cast<uint32_t>(info.tile_mode)} << 32u) ^ (uint64_t {static_cast<uint32_t>(info.guest_format)} << 48u);
}
static bool MismaForma(const ImageInfo& a, const ImageInfo& b) {
	return a.pixel_format == b.pixel_format && a.guest_format == b.guest_format && a.tile_mode == b.tile_mode &&
	       a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.extent.depth == b.extent.depth &&
	       a.resources.levels == b.resources.levels && a.resources.layers == b.resources.layers && a.type == b.type &&
	       a.data.size == b.data.size && a.samples == b.samples;
}
} // namespace TexDedup

// (M7, GestorMem) Conexion de la cache de texturas con el gestor central (src/local/gestor-memoria.h): cada imagen
// registrada (RegisterImage) es un recurso de VRAM con su tipo (textura / RT / profundidad) y FlagRespaldo (se rehace desde
// la RAM del juego) o FlagSoloGpu (escrita por la GPU, o su RAM soltada por VRAM_UNICA, o bytes nativos solo en la GPU).
// Tocar en TouchImage, Liberar en UnregisterImage (DeleteImage y todas las expulsiones pasan por ahi). Las dispersas /
// nativas las registra sparseImage.cpp (TipoNativa, VRAM ligada). Expulsor de texturas: el gestor (hilo de render, en
// RefreshMemoryBudget) deja sus candidatas en una cola y GestorPass (RunGarbageCollector, con m_lock) las borra: nunca se
// toma m_lock desde el gestor (TrackingSpinLock no es recursivo ni tiene try_lock). RAM: un recurso "copias RAM" (Lugar
// Ram) con los bytes de RAM del juego de texturas que ya estan en la VRAM (lo que el offload / VRAM_UNICA puede soltar);
// su expulsor, con la RAM en critico, fuerza la pasada del offload (OffloadScan) aunque no toque.
namespace GestorTex {
static std::mutex                           cola_m;
static std::vector<GestorMem::Candidato>    cola;      // bajo cola_m
static std::atomic<bool>                    ram_prisa {false};
static bool                                 forzar_scan = false; // (hilo de RunGarbageCollector)
static uint64_t                             copias_id   = 0;
static std::atomic<uint64_t>                c_n {0}, c_bytes {0}, c_saltadas {0};
static void* Dueno(ImageId id) {
	return reinterpret_cast<void*>((uint64_t {id.generation} << 32u) | id.index);
}
static ImageId DeDueno(void* d) {
	const auto v = reinterpret_cast<uint64_t>(d);
	return ImageId(static_cast<uint32_t>(v & 0xffffffffu), static_cast<uint32_t>(v >> 32u));
}
static int Tipo(const Image& im) {
	if (im.IsSparse()) return GestorMem::TipoNativa;
	if (im.info.IsDepth() || im.info.HasStencil() || im.usage.depth_target || im.depth_id) return GestorMem::TipoProfundidad;
	if (im.usage.render_target || im.binding.is_target || im.usage.storage || im.usage.video_out) return GestorMem::TipoRenderTarget;
	return GestorMem::TipoTextura;
}
static uint32_t Flags(const Image& im, int tipo) {
	if (tipo != GestorMem::TipoTextura || im.IsGpuModified()) return GestorMem::FlagSoloGpu;
	if (TexNativa::On() && TexNativa::AnyOwned(im.info.data.address, im.info.data.End())) return GestorMem::FlagSoloGpu;
	if (TextureOffload::Overlaps(im.info.data.address, im.info.data.size)) return GestorMem::FlagSoloGpu; // RAM soltada
	return GestorMem::FlagRespaldo;
}
static void Reclasifica(Image& im) {
	if (im.gestor_id == 0) return;
	const int      t = Tipo(im);
	const uint32_t f = Flags(im, t);
	if (t == im.gestor_tipo && f == im.gestor_flags) return;
	im.gestor_tipo  = static_cast<int8_t>(t);
	im.gestor_flags = static_cast<uint8_t>(f);
	GestorMem::CambiarTipo(im.gestor_id, t, f);
}
// Hilo del gestor (RefreshMemoryBudget), sin locks de la cache: solo encola.
static uint64_t ExpulsorTexturas(void* /*ctx*/, const GestorMem::Candidato* c, size_t n, uint64_t objetivo) {
	uint64_t        sum = 0;
	std::lock_guard lk(cola_m);
	if (cola.size() > 4096) return 0; // la pasada anterior aun no se consumio
	for (size_t i = 0; i < n && sum < objetivo; i++) {
		if (c[i].accion != GestorMem::AccionExpulsar) continue; // sin respaldo: nunca se borra
		cola.push_back(c[i]);
		sum += c[i].bytes;
	}
	return sum; // se borran en el siguiente RunGarbageCollector (asincrono, como el juego con sus mips)
}
static uint64_t ExpulsorCopiasRam(void* /*ctx*/, const GestorMem::Candidato* /*c*/, size_t /*n*/, uint64_t /*objetivo*/) {
	if (GestorMem::Critica(GestorMem::Ram)) ram_prisa.store(true, std::memory_order_relaxed);
	return 0; // lo suelta OffloadScan en el siguiente RunGarbageCollector
}
// ram-guard (su hilo, cada segundo): con presion critica de RAM, la misma pasada de copias RAM de texturas.
static uint64_t FlushRamGuard(int level) {
	if (level >= RamGuard::Critical) ram_prisa.store(true, std::memory_order_relaxed);
	return 0;
}
} // namespace GestorTex

ImageId TextureCache::InsertImage(const ImageInfo& info, bool may_fail) {
	VramReparto::Created(info.data.address); // (VRAM-REPARTO)
	const Envio1::Scope envio1_new(Envio1::TexNew); // (envio1)
	auto id = m_slot_images.insert(m_graphics, m_scheduler, info, may_fail);
	if (info.resources.layers > 1 && info.data.size >= (uint64_t {256} << 20u)) { // (vram8: the [vram8] line)
		Vram8::big_n.fetch_add(1, std::memory_order_relaxed);
		Vram8::big_bytes.fetch_add(info.data.size, std::memory_order_relaxed);
	}
	// An image that must exist (a render target, a depth image, a copy's) the GPU could not make: room is made at once
	// (the second time all there is, the images the GPU wrote too) and it is made again; then again for a while, as a
	// buffer is (GraphicContext::RetryAllocation); failing still ends the emulator, as it did at once before.
	// Issue #53: a 16 GB RX 7800 XT whose video memory another program had taken switched to the cinematic mode (AMD
	// gives these images no system memory). Issue #252: an 8 GB RX 6650 XT with FSR 3 frame generation at 3840x1200,
	// 1.3 GB over its budget, freed 143 MiB of textures for a 135 MiB target and still had no room: the textures freed
	// sat between others in VMA's 64 MiB blocks (2 of 121 memory objects went), and the target needs its own.
	const bool test = !may_fail && TestReclaim(info, m_gc_tick);
	if (!may_fail && (m_slot_images[id].CreateFailed() || test)) {
		if (!test) m_failed_tick = m_gc_tick + 1; // (the collector takes more for a while: RunGarbageCollector)
		for (uint32_t round = 0;; ++round) {
			RetireImage(id); // (never registered)
			(void)ReclaimImageMemory(info.data.size, round != 0);
			id = m_slot_images.insert(m_graphics, m_scheduler, info, false);
			if (!m_slot_images[id].CreateFailed() && !(test && round + 1 < TestReclaimRounds())) break;
			if (round == 0) continue;
			// (Port: the fork's GraphicContext::RetryAllocation inline: every 100 ms for up to 10 s, the images kept for
			// reuse freed each time.)
			bool made = false;
			for (int attempt = 0; attempt < 100 && !made; attempt++) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				m_graphics.TrimImagePool();
				RetireImage(id); // (never registered)
				id   = m_slot_images.insert(m_graphics, m_scheduler, info, false);
				made = !m_slot_images[id].CreateFailed();
			}
			std::printf("Texture cache: a %llu MiB image %s after retrying\n", static_cast<unsigned long long>(info.data.size >> 20u),
			            made ? "made" : "still not made");
			std::fflush(stdout);
			if (made) break;
			EXIT("failed to create image: extent=%ux%ux%u format=%d layers=%u levels=%u: no memory for it\n",
			     info.extent.width, info.extent.height, info.extent.depth, static_cast<int>(info.pixel_format),
			     info.resources.layers, info.resources.levels);
		}
	}
	if (m_slot_images[id].CreateFailed()) {
		RetireImage(id); // (never registered)
		if (m_failed_images.size() >= 16) m_failed_images.erase(m_failed_images.begin());
		m_failed_images.push_back({info.data.address, info.data.size, m_gc_tick});
		m_failed_tick = m_gc_tick + 1;
		return {};
	}
	if (!info.data.Empty()) {
		RegisterImage(id);
		RamNativeTex::ProbeEnd(info.data.address, info.data.End(), true); // (ram1b: the first-use probe)
	}
	return id;
}

// The images no draw or dispatch used for 16 flips whose bytes guest memory holds (none the GPU wrote: one used again is
// made again from them) go at once, least recently used first, until twice the room asked for (at least 256 MiB) is
// freed (`all`: every one, and those the GPU wrote, their bytes downloaded to guest memory first, as the collector takes
// them while memory is short: RunGarbageCollector); the GPU finishes the work in flight first (it may read them still).
// (The collector frees images unused for 30 flips at the earliest, and a mode switch needs the room in the flip it makes
// its targets. A table set touches its images at least every 8 flips: TableImageSet::TouchTicks.)
static size_t NativeEvictCap(size_t normal) {
	// (log 051042: cuelgue al entrar en la niebla, eboot+0xa6f235, puntero nulo en un objeto del juego, con hasta 8
	// expulsiones nativas por pasada: restaurar a la RAM puede escribir sobre memoria que el juego ya reutilizo por otra via
	// que no es free 0x802f90. Apagado por defecto; KYTY_TEX_NATIVA_EXPULSAR=N lo permite.)
	static const long cap = [] {
		const char* v = std::getenv("KYTY_TEX_NATIVA_EXPULSAR");
		return v != nullptr ? std::atol(v) : 0L;
	}();
	return cap < 0 ? normal : std::min<size_t>(normal, static_cast<size_t>(cap));
}

size_t TextureCache::ReclaimImageMemory(uint64_t bytes, bool all) {
	constexpr uint64_t UnusedTicks = 16;
	// (A speculation's recorder cannot wait for the GPU: the GPU thread translates that work again.)
	if (CommandScheduler::ThreadRecorder() != nullptr) throw CommandScheduler::RecorderRefusal {"ReclaimImageMemory"};
	const auto           start        = std::chrono::steady_clock::now();
	const uint64_t       goal         = all ? UINT64_MAX : std::max(bytes * 2, uint64_t {256} << 20u);
	const uint64_t       download_max = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download).Size() / 2;
	uint64_t             freed = 0, written_bytes = 0;
	size_t               written = 0;
	std::vector<ImageId> victims;
	std::vector<ImageId> native_later; // (P1)
	// (KYTY_CUBE_SHADOW_FIX: a shadow pool's parked twin goes first; the next switch makes it again.)
	std::vector<ImageId> twins;
	std::erase_if(m_parked, [&](const auto& item) {
		if (item.second.keep <= 1 || freed >= goal) return false;
		if (const auto* twin = m_slot_images.try_get(item.second.id)) freed += twin->AccountedSize();
		twins.push_back(item.second.id);
		return true;
	});
	// (N3-BLOQ) Nivel 3 bloqueado por inutil: sin `all` no se expulsa nada por este camino tampoco.
	const bool n3_bloq = !all && VramTope::On() && VramTope::g_n3_bloqueado.load(std::memory_order_relaxed);
	if (m_gc_tick >= UnusedTicks && !n3_bloq) {
		m_lru_cache.ForEachItemBelow(m_gc_tick - UnusedTicks, [&](ImageId id) {
			const auto* owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) return false;
			if (!all && VramTope::On() && GcProtegida(*owner)) return false; // (N3-REAL)
			// (N3-BLOQ) Nativa usada en los ultimos ~30 s: se queda (volveria enseguida).
			if (!all && VramTope::On() && (owner->IsSparse() || TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End())) &&
			    m_lru_cache.TickOf(owner->lru_id) + VramTope::NatUsoTicks > m_gc_tick)
				return false;
			// (P1) Con rangos "solo en la GPU": al final, como mucho 2 (abajo).
			if (TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End())) {
				if (native_later.size() < NativeEvictCap(2)) native_later.push_back(id);
				return false;
			}
			// (A tiled one too large for the download buffer stays, as in the collector.)
			if (owner->IsGpuModified() && (!all || (owner->info.IsTiled() && SafeToDownload(*owner) &&
			                                        owner->info.data.size > download_max)))
				return false;
			if (!all && GcGpuSuciaRetiene(*owner)) return false; // (GC-GPU-SUCIA)
			victims.push_back(id);
			freed += owner->AccountedSize();
			return freed >= goal;
		});
	}
	// (P1) Solo si aun falta: recuperar sus bytes a la RAM ANTES de DetachImage; si no se puede, no se expulsa.
	for (const auto id: native_later) {
		if (freed >= goal) break;
		auto* owner = m_slot_images.try_get(id);
		if (owner == nullptr || !owner->registered || !NativeRestoreBeforeDelete(*owner, true)) continue;
		owner = m_slot_images.try_get(id); // (FlushAndWait puede invalidar el puntero)
		if (owner == nullptr || !owner->registered) continue;
		if (TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End())) continue;
		victims.push_back(id);
		freed += owner->AccountedSize();
	}
	// (One the GPU wrote whose download did not start stays. Not safe to download: a GPU buffer write is newer.)
	std::erase_if(victims, [&](ImageId id) {
		auto& owner = m_slot_images[id];
		if (!owner.IsGpuModified()) return false;
		if (SafeToDownload(owner) && !TryDownloadImage(id)) {
			freed -= owner.AccountedSize();
			return true;
		}
		if (!SafeToDownload(owner)) GcGpuSuciaPerdida(owner); // (GC-GPU-SUCIA) borrada sin descarga
		owner.ClearGpuModified();
		written++;
		written_bytes += owner.AccountedSize();
		return false;
	});
	std::erase_if(victims, [&](ImageId id) { return !DetachImage(id); });
	for (const auto id: twins) victims.push_back(id); // (parked: already out of every index)
	if (!victims.empty()) {
		m_scheduler.Wait(m_scheduler.CurrentTick());
		// (Destroyed now, not after RetireImage's operation: the GPU is done with them.)
		const auto packet = Spec::PacketNow();
		for (const auto id: victims) m_retired.emplace_back(id, packet);
		EraseRetired();
		DrainImageDestroys(); // (DESTRUIR-HILO, vma.cpp: the caller makes its image in the room freed here)
		LiveCounters::Add(LiveCounters::GcImageDeletes, victims.size());
		VramSaverAdd(VramEvicted, victims.size());
		VramSaverAdd(VramEvictedBytes, freed);
	}
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("Texture cache: no memory for a %llu MiB image, %zu images unused for %llu flips freed (%llu MiB; %zu "
		            "of them the GPU wrote, %llu MiB) in %.1f ms\n",
		            static_cast<unsigned long long>(bytes >> 20u), victims.size(),
		            static_cast<unsigned long long>(UnusedTicks), static_cast<unsigned long long>(freed >> 20u), written,
		            static_cast<unsigned long long>(written_bytes >> 20u),
		            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
		std::fflush(stdout);
	}
	return victims.size();
}

bool TextureCache::RecentlyFailed(const ImageInfo& info) const {
	return std::ranges::any_of(m_failed_images, [&](const FailedImage& failed) {
		return failed.address == info.data.address && info.data.size >= failed.size &&
		       m_gc_tick < failed.tick + FailedRetryTicks;
	});
}

void TextureCache::StampRegistrationPages(const Image& image, uint64_t epoch) {
	if (m_registration_log.size() >= 8192) m_registration_log.erase(m_registration_log.begin(), m_registration_log.begin() + 4096);
	m_registration_log.push_back({epoch, image.info.data.address, image.info.data.End()});
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) return;
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) m_registration_pages[page] = epoch;
}

bool TextureCache::RegistrationsOverlapSince(uint64_t address, uint64_t size, uint64_t epoch, bool& known) const {
	const auto& log = m_registration_log;
	known           = log.empty() || log.front().epoch <= epoch + 1;
	if (!known) return true;
	// (At the stamps' page size: what RegistrationsSince would see.)
	constexpr uint64_t page  = uint64_t {1} << ImageEpochTable::kPageBits;
	const auto         begin = address & ~(page - 1), end = (address + size + page - 1) & ~(page - 1);
	for (auto entry = log.rbegin(); entry != log.rend() && entry->epoch > epoch; ++entry)
		if ((entry->begin & ~(page - 1)) < end && begin < ((entry->end + page - 1) & ~(page - 1))) return true;
	return false;
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

// (11-25c, KYTY_TEX_NATIVA) Base del T# de una CTexture -> sus imagenes nativas (dispersas por mip) registradas, la ultima
// al final. Bajo m_lock. (11-25d, revision 7) Lista: dos imagenes en la misma base (p. ej. la vieja aun registrada al crear
// la nueva) ya no se pisan; al quitar una, la otra sigue encontrandose.
static std::unordered_map<uint64_t, std::vector<ImageId>>& NativeByBase() {
	static std::unordered_map<uint64_t, std::vector<ImageId>> m;
	return m;
}
// (11-25d, revision 4) Imagenes con bytes "solo en la GPU" que se borraron (DeleteImage) o aparcaron (ParkImage) donde no
// se puede descargar (grabador / operacion diferida): ya fuera del registro, su imagen de Vulkan sigue viva aqui hasta que
// NativeRingCollect (hilo de render, antes de un draw) recupera sus bytes a la RAM y la retira. Antes se perdian
// (RemoveOwned sin copia: textura vieja o corrupta al recrearla). Bajo m_lock.
// (WB-EPOCA) Con la epoca del registro al sacarla del registro: lo que cambie despues (liberado, reutilizado o de otra
// textura nueva en la misma memoria) no se escribe con sus bytes viejos.
static std::vector<std::pair<ImageId, uint64_t>>& NativeRestoreLater() {
	static std::vector<std::pair<ImageId, uint64_t>> v;
	return v;
}
static std::atomic<bool>     g_native_restore_later {false};
// (contadores: TexNativa::c_later_n / c_later_ok)

bool TextureCache::IsNativeSparse(const Image& image) const {
	return SparseIsPerLevel(image.backing);
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
		if (const auto bucket = StartBucket(image.info.data.address); m_start_buckets[bucket]++ == 0)
			m_start_bucket_bits[bucket / 64u] |= uint64_t {1} << (bucket % 64u);
		const auto epoch = m_start_epoch.load(std::memory_order_relaxed) + 1;
		if (m_start_log.size() >= 8192) m_start_log.erase(m_start_log.begin(), m_start_log.begin() + 4096);
		m_start_log.emplace_back(epoch, image.info.data.address);
		m_start_epoch.store(epoch, std::memory_order_release);
	}
	TexDirecta::MarkImage(image.info.data.address, image.info.data.size, +1); // (etapa 2.1c) mapa sin bloqueo para el memcpy
	if (image.registrations == 0) image.neg_origen = TexNativa::t_neg_origen; // (NEGRAS-LOG)
	if (TexNativa::On() && SparseIsPerLevel(image.backing) && TexNativa::IsNative(image.info.data.address)) {
		auto& list = NativeByBase()[image.info.data.address];
		list.push_back(id); // (11-25c) CTexture -> su imagen nativa
		// (NEGRAS-LOG b) Dos imagenes nativas con la misma base: los trozos solo van a la mas reciente.
		if (TexNativa::NegrasLog() && list.size() > 1 && TexNativa::NegrasLinea(1)) {
			std::printf("[negra] doble-nativa base=0x%llx n=%zu:", static_cast<unsigned long long>(image.info.data.address),
			            list.size());
			for (const auto other: list) {
				const auto* o = m_slot_images.try_get(other);
				if (o == nullptr) continue;
				std::printf(" {serial=%llu reg=%d origen=%s fmt=%u vk=%u %ux%u niveles=%u size=0x%llx}",
				            static_cast<unsigned long long>(o->serial), (o->registered || other == id) ? 1 : 0,
				            TexNativa::NegOrigen(other == id ? TexNativa::t_neg_origen : o->neg_origen),
				            static_cast<uint32_t>(o->info.guest_format), static_cast<uint32_t>(o->info.pixel_format),
				            o->info.extent.width, o->info.extent.height, o->info.resources.levels,
				            static_cast<unsigned long long>(o->info.data.size));
			}
			std::printf("\n");
			std::fflush(stdout);
		}
	}
	image.registered = true;
	++image.registrations;
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	m_total_used_memory += image.AccountedSize();
	if (!image.IsSparse()) { // (M7) al gestor central; las dispersas/nativas las registra sparseImage.cpp (VRAM ligada)
		const int      t  = GestorTex::Tipo(image);
		const uint32_t f  = GestorTex::Flags(image, t);
		image.gestor_tipo  = static_cast<int8_t>(t);
		image.gestor_flags = static_cast<uint8_t>(f);
		// Un alias de la VkImage de otra (TEX-DEDUP) no ocupa VRAM propia: 0 bytes.
		image.gestor_id = GestorMem::Registrar(t, GestorMem::Vram, (image.SharedWithOthers() || image.MemAliasGuest()) ? 0 : image.AccountedSize(), f,
		                                       GestorTex::Dueno(id));
	}
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	// Its dropped guest bytes back first (texture-offload.h).
	if (TextureOffload::GetMode() == TextureOffload::Mode::Offload && g_offload_alive.load(std::memory_order_relaxed) &&
	    TextureOffload::Overlaps(image.info.data.address, image.info.data.size)) {
		// (Its download goes with the work before the image is erased; the bytes land when the GPU is done.)
		RestoreOffloaded(image.info.data.address, image.info.data.size, static_cast<uint8_t>(TextureOffload::Touch::ImageGone),
		                 true);
	}
	TexDirecta::MarkImage(image.info.data.address, image.info.data.size, -1); // (etapa 2.1c)
	if (TexNativa::On()) { // (11-25c)
		auto& by_base = NativeByBase();
		if (auto it = by_base.find(image.info.data.address); it != by_base.end()) {
			std::erase_if(it->second, [&](ImageId x) { return x == id; });
			if (it->second.empty()) by_base.erase(it);
		}
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
		if (const auto bucket = StartBucket(image.info.data.address); --m_start_buckets[bucket] == 0)
			m_start_bucket_bits[bucket / 64u] &= ~(uint64_t {1} << (bucket % 64u));
	}
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
	GestorMem::Liberar(image.gestor_id); // (M7)
	image.gestor_id   = 0;
	image.gestor_tipo = -1;
}

void TextureCache::DeleteImage(ImageId id) {
	// (N2+N3c) Una imagen con bytes "solo en la GPU" que se borra con su memoria aun comprometida: esos bytes se descargan
	// de la GPU y se escriben en la RAM emulada antes (NativeRestoreBeforeDelete). (N2+N3g) Los recolectores bajo presion
	// la recuperan ellos antes (keep_on_fail) y, si no se puede, no la borran.
	if (auto* image = m_slot_images.try_get(id); image != nullptr && image->registered && TexNativa::On()) {
		// (11-25d, revision 4) Desde el grabador o una operacion diferida no se puede descargar: fuera del registro ya, y
		// sus bytes a la RAM despues (NativeRestoreLater), no perdidos.
		if ((CommandScheduler::ThreadRecorder() != nullptr || CommandScheduler::InDeferredOperation()) && !image->depth_id &&
		    TexNativa::AnyOwned(image->info.data.address, image->info.data.End())) {
			if (DetachImage(id)) {
				NativeRestoreLater().emplace_back(id, TexNativa::EpocaActual());
				TexNativa::c_later_n.fetch_add(1, std::memory_order_relaxed);
				g_native_restore_later.store(true, std::memory_order_release);
			}
			return;
		}
		NativeRestoreBeforeDelete(*image);
	}
	if (DetachImage(id)) RetireImage(id);
}

bool TextureCache::DetachImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return false;
	}
	m_partial_plans.erase(image->serial);
	m_layer_owners.erase(image->serial);
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
	return true;
}

void TextureCache::RetireImage(ImageId id) {
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

// The guest's bytes are all of the image: only sampled (no GPU write, target, storage or video-out use), without
// metadata, stencil or depth. Another view of its memory can be another image: each refreshes from those bytes (as
// the buffer cache holds them), a CPU write invalidates both (InvalidateCpuAliases), a GPU buffer write marks both.
bool TextureCache::GuestTexture(const Image& image) {
	return image.registered && !image.depth_id && !image.IsGpuModified() && !image.usage.storage &&
	       !image.usage.render_target && !image.usage.depth_target && !image.usage.video_out && !image.binding.is_target &&
	       !image.info.IsDepth() && !image.info.HasMetadata() && !image.info.HasStencil() && image.info.samples == 1;
}

bool TextureCache::GuestTextureRequest(BindingType type, const ImageInfo& info) {
	return type == BindingType::Texture && !info.IsDepth() && !info.HasMetadata() && !info.HasStencil() &&
	       info.samples == 1;
}

void TextureCache::NoteGuestAlias(const ImageInfo& requested, Image& kept) {
	++m_guest_aliases;
	kept.guest_alias = true;
	m_alias_insert   = true; // (FindImage marks the image it inserts)
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("TextureCache: guest texture alias addr=0x%llx size=0x%llx levels=%u layers=%u next to size=0x%llx "
		            "levels=%u layers=%u fmt=%u %ux%u\n",
		            static_cast<unsigned long long>(requested.data.address),
		            static_cast<unsigned long long>(requested.data.size), requested.resources.levels,
		            requested.resources.layers, static_cast<unsigned long long>(kept.info.data.size),
		            kept.info.resources.levels, kept.info.resources.layers, static_cast<uint32_t>(requested.guest_format),
		            requested.extent.width, requested.extent.height);
		std::fflush(stdout);
	}
}

// An image a write binding found: the guest-texture aliases at its address go (written, it no longer holds what they
// read from the guest's bytes).
void TextureCache::FreeGuestAliases(ImageId written) {
	const auto                address = m_slot_images[written].info.data.address;
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, 1, pages)) return;
	ImageIds others;
	if (const auto* owners = m_image_page_table.Find(pages.first)) {
		owners->ForEach([&](ImageId id) {
			const auto* image = m_slot_images.try_get(id);
			if (id != written && image != nullptr && image->guest_alias && image->info.data.address == address &&
			    GuestTexture(*image)) {
				others.push_back(id);
			}
		});
	}
	for (const auto id: others) FreeImage(id, "guest-alias-written");
}

void TextureCache::TouchImage(Image& image) {
	GestorMem::Tocar(image.gestor_id); // (M7) sin locks
	if (auto* spec = Spec::Current()) { // (touched when the speculative translation is committed)
		auto& touched = spec->touched_images;
		if (touched.empty() || touched.back().first != &image) touched.emplace_back(&image, image.serial);
		return;
	}
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
	}
	if (image.IsSparse()) SparseMoverCancelar(image.backing); // (W14, SPARSE-COMPACTAR) se usa: no se traslada ahora
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	if (image.NeedsMaybeCpuHash()) {
		image.SetMaybeCpuHash(image.HashGuestEdges());
	}
	UntrackImage(id, "maybe-dirty");
}

namespace {

// KYTY_ASYNC_REPROTECT: the write protection of the pages TrackImage starts watching (again: the pages a refresh
// released) goes to the upload worker ahead of the upload's copies that read them whole (BufferCache::ImageProtectSink),
// is applied before any other read of them, and when the upload returns where nothing read them. In the frames that
// stream textures in, those protection calls took milliseconds of the render thread.
class DeferredImageProtects {
public:
	DeferredImageProtects(BufferCache& cache, PageManager& manager): m_cache(cache), m_manager(manager) {}
	DeferredImageProtects(const DeferredImageProtects&)            = delete;
	DeferredImageProtects& operator=(const DeferredImageProtects&) = delete;
	~DeferredImageProtects() { m_cache.ApplyImageProtects(); }

	template <class F>
	void Track(F&& track) {
		const bool defer = kyty_local_async_reprotect_mode.load(std::memory_order_relaxed) != 0;
		if (defer) PageManager::SetDeferredWriteProtectSink(m_cache.ImageProtectSink(m_manager));
		track();
		if (defer) PageManager::SetDeferredWriteProtectSink(nullptr);
	}

private:
	BufferCache& m_cache;
	PageManager& m_manager;
};

} // namespace

namespace {

// (P4) Las texturas nativas no vigilan sus paginas: sus trozos llegan por el memcpy interceptado (o kRamWrite) y
// la CPU del juego no escribe ahi por otro camino (Ghidra: 0x8fed14 solo antes de crear la imagen). El estado
// logico (track_addr, huecos) no cambia; solo desaparece el VirtualProtect. Los contadores son por pagina, asi
// que una imagen no nativa que comparta paginas sigue protegiendolas. Nativa es fijo en toda la vida de la imagen.
// (RT-NATIVO) KYTY_RT_NATIVO (por defecto 1; =0 como antes; necesita KYTY_MEMMAP para conocer las arenas): una imagen
// cuya memoria cae entera en RTSharedAllocator o RTPersistAllocator (Loader::DemonsSoulsMemMap::RtArenaOf) vive solo en
// la GPU: sin vigilancia de paginas (ni VirtualProtect), fuera del offload (sin vigilancia de lectura) y su copia de RAM
// queda reservada sin tocar. Las descargas que el cache ya hace bajo demanda (expulsion, solape/alias, lectura por un
// buffer de la GPU) siguen igual. Ghidra 01.007.000 (ver DsRtOffloadMode): el motor reserva las dos arenas enteras al
// arrancar solo para la GPU (PS5CRenderDisplay 0x90ad80). La decision se toma al primer WatchImagePages de la imagen y
// queda fija toda su vida (por serial): asi cada <true> tiene su <false> igual y los contadores por pagina no se
// descuadran. Las imagenes vigiladas antes de que el mapa lea las arenas siguen siendo normales.
namespace RtNativo {
inline bool On() {
	static const bool on = [] {
		const char* v  = std::getenv("KYTY_RT_NATIVO");
		const bool  ok = v != nullptr && (v[0] == '1' || v[0] == '2'); // (11-26 limpieza) apagado por defecto
		std::printf("RT nativo: %s (KYTY_RT_NATIVO=%s): imagenes en RTShared/RTPersist sin vigilancia de paginas ni offload "
		            "(necesita KYTY_MEMMAP)\n",
		            ok ? "on" : "off", v != nullptr ? v : "1 por defecto");
		std::fflush(stdout);
		return ok;
	}();
	return on;
}
inline std::mutex&                   Mutex() { static std::mutex m; return m; }
inline std::unordered_set<uint64_t>& Set() { static std::unordered_set<uint64_t> s; return s; } // seriales nativos
inline std::atomic<uint64_t> c_evitadas {0}, c_evitadas_bytes {0}, c_descargas {0}, c_descargas_bytes {0}, c_subidas {0},
    c_subidas_bytes {0}, c_altas {0};
inline bool Es(const Image& image) {
	if (!On()) return false;
	std::lock_guard<std::mutex> lock(Mutex());
	return Set().contains(image.serial);
}
// Al primer TrackImage completo de la imagen (track_addr recien puesto, aun sin paginas vigiladas).
inline void Alta(const Image& image) {
	if (!On() || image.info.data.Empty()) return;
	if (Loader::DemonsSoulsMemMap::RtArenaOf(image.info.data.address, image.info.data.End()) == 0) return;
	std::lock_guard<std::mutex> lock(Mutex());
	if (Set().insert(image.serial).second) c_altas.fetch_add(1, std::memory_order_relaxed);
}
inline void Hereda(const Image& from, const Image& to) { // (sustituta que toma la vigilancia de la otra)
	if (!On()) return;
	std::lock_guard<std::mutex> lock(Mutex());
	if (Set().contains(from.serial)) Set().insert(to.serial);
}
inline void Baja(uint64_t serial) {
	if (!On()) return;
	std::lock_guard<std::mutex> lock(Mutex());
	Set().erase(serial);
}
// (RT-NATIVO2) KYTY_RT_NATIVO=2: fase 2. Ghidra 01.007.000: PS5CTexture create 0x8ffed0 / 0x8ff080 piden la memoria
// al asignador (0x7f9cf0 / 0x7f9f70) e inicializan el descriptor (sce::Agc::Core::initialize); ninguna escritura de
// pixeles por la CPU. Asi la RAM de un RT en la arena solo cambia si el emulador descarga ahi algo de la GPU. Mientras
// ningun bloque de su rango se haya descargado nunca, la primera subida (solo CpuDirty, sin escrituras por buffer de la
// GPU) leeria los ceros de la reserva: se hace un clear a 0 en la GPU, sin leer ni tocar la RAM.
inline bool Fase2() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RT_NATIVO");
		return v != nullptr && v[0] == '2'; // (11-26 limpieza: apagado por defecto) (antes por defecto fase 2; =1 solo fase 1; =0 nada)
	}();
	return on;
}
constexpr uint64_t                   kBloque = 64 * 1024;
inline std::unordered_set<uint64_t>& Sucios() { static std::unordered_set<uint64_t> s; return s; } // bloques descargados
inline std::atomic<uint64_t> c_clears {0}, c_clears_bytes {0}, c_sub_cpu {0}, c_sub_buffer {0};
inline void MarcaDescarga(uint64_t address, uint64_t size) { // toda descarga a RAM (de cualquier imagen)
	if (!On() || !Fase2() || size == 0) return;
	std::lock_guard<std::mutex> lock(Mutex());
	for (uint64_t b = address / kBloque; b <= (address + size - 1) / kBloque; b++) Sucios().insert(b);
}
inline bool RamVirgen(uint64_t address, uint64_t size) {
	std::lock_guard<std::mutex> lock(Mutex());
	if (Sucios().empty()) return true;
	for (uint64_t b = address / kBloque; b <= (address + size - 1) / kBloque; b++)
		if (Sucios().contains(b)) return false;
	return true;
}
inline void Informe() { // cada 30 s
	if (!On()) return;
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last = Clock::now();
	const auto               now  = Clock::now();
	if (now - last < std::chrono::seconds(30)) return;
	last = now;
	size_t vivas = 0;
	{
		std::lock_guard<std::mutex> lock(Mutex());
		vivas = Set().size();
	}
	constexpr double MiB = 1024.0 * 1024.0;
	if (Fase2()) { // (RT-NATIVO2)
		const auto clears = c_clears.exchange(0), sub_cpu = c_sub_cpu.exchange(0), sub_buf = c_sub_buffer.exchange(0);
		const auto total  = clears + sub_cpu + sub_buf;
		std::printf("[rt-nativo2] (30 s) primeras subidas nativas (clear GPU, 0 bytes de RAM) %llu (%.0f MiB) = %.1f%%; "
		            "subidas RAM por CPU/descarga previa %llu; por escritura GPU via buffer %llu\n",
		            static_cast<unsigned long long>(clears), static_cast<double>(c_clears_bytes.exchange(0)) / MiB,
		            total ? 100.0 * static_cast<double>(clears) / static_cast<double>(total) : 100.0,
		            static_cast<unsigned long long>(sub_cpu), static_cast<unsigned long long>(sub_buf));
	}
	std::printf("[rt-nativo] (30 s) imagenes RT nativas vivas %zu (nuevas %llu); protecciones evitadas %llu (%.0f MiB); "
	            "descargas bajo demanda %llu (%.0f MiB); subidas desde RAM %llu (%.0f MiB)\n",
	            vivas, static_cast<unsigned long long>(c_altas.exchange(0)), static_cast<unsigned long long>(c_evitadas.exchange(0)),
	            static_cast<double>(c_evitadas_bytes.exchange(0)) / MiB, static_cast<unsigned long long>(c_descargas.exchange(0)),
	            static_cast<double>(c_descargas_bytes.exchange(0)) / MiB, static_cast<unsigned long long>(c_subidas.exchange(0)),
	            static_cast<double>(c_subidas_bytes.exchange(0)) / MiB);
	std::fflush(stdout);
}
} // namespace RtNativo

// (GPUNAT-RT, n2 11-25) KYTY_GPUNAT_RT_RAM (por defecto 1; =0 como antes). Un RT/profundidad de las arenas del juego
// (RTSharedAllocator / RTPersistAllocator, ya "RT nativo": sin vigilancia de paginas) vive solo en la VRAM, como en un
// juego de PC: su copia en la RAM del juego se suelta (MEM_RESET + fuera del working set, NativeOffloadDrop), una vez por
// imagen. Ghidra 01.007.000: 0x91c680 (RenderTargetManager::InitSharedMemory) crea las dos arenas lineales con
// 0x801d00 (alineadas a 64 KiB); su Reset 0x915930 solo vuelve el cursor (+0xb0 = +0x90): la CPU nunca escribe ni lee
// pixeles ahi (PS5CTexture create 0x8ffed0 / 0x8ff080 solo rellenan el descriptor). Medido (log 20261125-202747-644,
// [ram19]): RTShared 0,16 GB en RAM (0,09 con copia en la GPU), RTPersist 0,07 (0,05): RAM que solo es copia vieja.
// Condiciones (todas en el hilo de la GPU con m_lock, la pasada de RunGarbageCollector): la imagen y TODAS las que
// solapan sus paginas son RT nativos con la GPU como duena (SafeToDownload: escrita por la GPU, sin escrituras por buffer
// ni de la CPU pendientes); sin descarga pendiente; fuera del offload y de las texturas nativas; ningun buffer de la GPU
// con bytes sucios o con escrituras de la CPU sin subir en ese rango. Si despues algo la descarga (expulsion, solape) la
// RAM se vuelve a escribir como siempre. Un RT recreado que suba desde esa RAM leera ceros en vez de un fotograma viejo
// (el RT nativo fase 2 ya hace un clear en ese caso). Tope 128 MiB por pasada (cada 60 ticks) para no dar tirones.
// Linea "[gpunat] rt-ram" cada 30 s: VRAM, RAM (working set) y ms/frame de la ventana anterior a la primera suelta (antes)
// y de la ultima (ahora), y lo soltado.
namespace GpuNatRt {
inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_GPUNAT_RT_RAM");
		const bool  r = (v != nullptr && v[0] == '1') && RtNativo::On();
		std::printf("[gpunat] KYTY_GPUNAT_RT_RAM=%d: RT de las arenas del juego solo en VRAM (su copia en RAM se suelta)\n", r ? 1 : 0);
		std::fflush(stdout);
		return r;
	}();
	return on;
}
inline std::unordered_set<uint64_t>& Hechos() { static std::unordered_set<uint64_t> s; return s; } // seriales (m_lock)
inline uint64_t c_imgs = 0, c_soltado = 0, c_rechazado = 0, c_sin_commit = 0, c_saltadas = 0, c_us = 0, c_us_max = 0;
struct Ventana {
	uint64_t frames = 0, vram = 0, ram = 0;
	double   ms     = 0.0;
};
inline Ventana antes {}, ahora {};
inline bool    hay_antes = false;
// Cada frame (RunGarbageCollector): ms/frame, VRAM y RAM medidos por el gestor; linea cada 30 s.
inline void Frame() {
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last_frame = Clock::now(), last_log = Clock::now();
	static double            sum_ms = 0.0;
	static uint64_t          n = 0;
	const auto               now = Clock::now();
	sum_ms += std::chrono::duration<double, std::milli>(now - last_frame).count();
	n++;
	last_frame = now;
	if (now - last_log < std::chrono::seconds(30)) return;
	last_log = now;
	Ventana w;
	w.frames = n;
	w.ms     = n != 0 ? sum_ms / static_cast<double>(n) : 0.0;
	w.vram   = GestorMem::Uso(GestorMem::Vram) >> 20u;
	w.ram    = GestorMem::Uso(GestorMem::Ram) >> 20u;
	sum_ms = 0.0, n = 0;
	if (c_imgs == 0) antes = w, hay_antes = true; // (aun sin soltar nada: la ventana "antes")
	ahora = w;
	std::printf("[gpunat] rt-ram 30 s: ANTES (sin soltar) VRAM %llu MiB, RAM %llu MiB, %.2f ms/frame -> AHORA VRAM %llu MiB, "
	            "RAM %llu MiB, %.2f ms/frame; RT solo en VRAM %llu imagenes, RAM soltada %llu MiB (rechazada %llu, sin commit "
	            "%llu), saltadas por seguridad %llu; coste %.2f ms en total (max %.2f por pasada)\n",
	            static_cast<unsigned long long>(antes.vram), static_cast<unsigned long long>(antes.ram), antes.ms,
	            static_cast<unsigned long long>(w.vram), static_cast<unsigned long long>(w.ram), w.ms,
	            static_cast<unsigned long long>(c_imgs), static_cast<unsigned long long>(c_soltado >> 20u),
	            static_cast<unsigned long long>(c_rechazado >> 20u), static_cast<unsigned long long>(c_sin_commit >> 20u),
	            static_cast<unsigned long long>(c_saltadas), static_cast<double>(c_us) / 1000.0,
	            static_cast<double>(c_us_max) / 1000.0);
	std::fflush(stdout);
}
} // namespace GpuNatRt

template <bool track>
inline void WatchImagePages(PageManager& manager, const Image& image, uint64_t address, uint64_t size) {
	if (TexNativa::On() && SparseIsPerLevel(image.backing)) return;
	if (RtNativo::Es(image)) { // (RT-NATIVO) ni vigilancia ni VirtualProtect; track_addr/huecos siguen igual
		RtNativo::c_evitadas.fetch_add(1, std::memory_order_relaxed);
		RtNativo::c_evitadas_bytes.fetch_add(size, std::memory_order_relaxed);
		return;
	}
	manager.UpdatePageWatchers<track>(address, size);
}

} // namespace

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
		RtNativo::Alta(image); // (RT-NATIVO) sin paginas vigiladas aun: la decision vale para toda su vida
		WatchImagePages<true>(m_page_manager, image, image_begin, image.info.data.size);
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
	WatchImagePages<true>(m_page_manager, image, image_begin, size);
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
	WatchImagePages<true>(m_page_manager, image, address, size);
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
				WatchImagePages<false>(m_page_manager, image, cursor, hole_begin - cursor);
			}
			cursor = std::max(cursor, hole_end);
		}
		if (cursor < end) {
			WatchImagePages<false>(m_page_manager, image, cursor, end - cursor);
		}
		return;
	}
	if (size != 0) {
		WatchImagePages<false>(m_page_manager, image, address, size);
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
			WatchImagePages<false>(m_page_manager, image, cursor, hole_begin - cursor);
		}
		cursor = std::max(cursor, hole_end);
	}
	if (cursor < end) {
		WatchImagePages<false>(m_page_manager, image, cursor, end - cursor);
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
		WatchImagePages<true>(m_page_manager, image, hole_begin, hole_end - hole_begin);
	}
	image.untracked_holes.clear();
}

void TextureCache::RetrackHoles(Image& image, uint64_t begin, uint64_t end) {
	if (image.untracked_holes.empty()) {
		return;
	}
	if (!image.IsTracked()) {
		EXIT("TextureCache: untracked image keeps page holes\n");
	}
	std::vector<std::pair<uint64_t, uint64_t>> kept;
	kept.reserve(image.untracked_holes.size() + 1);
	for (const auto& [hole_begin, hole_end]: image.untracked_holes) {
		const auto from = std::max(hole_begin, begin);
		const auto to   = std::min(hole_end, end);
		if (from >= to) {
			kept.emplace_back(hole_begin, hole_end);
			continue;
		}
		WatchImagePages<true>(m_page_manager, image, from, to - from);
		if (hole_begin < from) kept.emplace_back(hole_begin, from);
		if (to < hole_end) kept.emplace_back(to, hole_end);
	}
	image.untracked_holes = std::move(kept);
}

void TextureCache::FinishRefresh(Image& image) {
	// Watch the released pages again before the image counts as clean.
	RetrackHoles(image);
	image.RefreshComplete();
}

// (M7, TEX-DEDUP) Bajo m_lock. La imagen del indice con el mismo contenido que `info` (hash de su RAM ahora), si sigue
// valiendo de fuente: viva, la misma (serial), solo muestreada y limpia (su VkImage == sus bytes del juego, vigilados),
// densa de VMA, con su RAM presente (no soltada por el offload), sin escrituras de bufer de la GPU y con su RAM, leida
// ahora, igual a la clave. Si no, sale del indice.
ImageId TextureCache::TexDedupFuente(const ImageInfo& info, uint64_t hash) {
	const auto it = TexDedup::indice.find(TexDedup::Clave(info, hash));
	if (it == TexDedup::indice.end()) return {};
	const auto* src = m_slot_images.try_get(it->second.id);
	const bool  ok  = src != nullptr && src->serial == it->second.serial && src->registered &&
	                src->info.data.address != info.data.address && TexDedup::MismaForma(src->info, info) && GuestTexture(*src) &&
	                !src->IsCpuDirty() && !src->IsBufferModified() && !src->IsStencilModified() && !src->IsSparse() &&
	                src->backing.image != nullptr && src->backing.allocation != nullptr && !src->share_fresh &&
	                !(TexNativa::On() && TexNativa::AnyOwned(src->info.data.address, src->info.data.End())) &&
	                !TextureOffload::Overlaps(src->info.data.address, src->info.data.size) &&
	                !m_buffer_cache.HasGpuDirtyBytes(src->info.data.address, src->info.data.size) &&
	                HashGuestRange(src->info.data.address, src->info.data.size) == hash;
	if (!ok) {
		TexDedup::indice.erase(it);
		return {};
	}
	return it->second.id;
}

// (M7, TEX-DEDUP) Bajo m_lock, justo tras InsertImage. Propia: al indice. Alias: se vigilan sus paginas y despues se
// comprueba otra vez su RAM (una escritura entre el hash y la vigilancia no se perderia); igual -> limpia sin subir nada
// (su contenido ya esta en la VkImage). Distinta -> fuera y otra imagen con su propia VkImage.
ImageId TextureCache::TexDedupAlta(ImageId id, const ImageInfo& info, uint64_t hash, bool may_fail) {
	auto& image = m_slot_images[id];
	if (!image.share_fresh) {
		if (image.backing.image != nullptr && !image.IsSparse()) {
			if (TexDedup::indice.size() >= 65536) TexDedup::indice.clear();
			TexDedup::indice[TexDedup::Clave(info, hash)] = TexDedup::Entrada {id, image.serial};
		}
		return id;
	}
	image.share_fresh = false;
	{
		DeferredImageProtects protects {m_buffer_cache, m_page_manager};
		protects.Track([&] { TrackImage(id); });
	}
	const bool igual = image.IsDefinitelyCpuDirty() && !image.IsMaybeCpuDirty() && !image.IsPartiallyCpuDirty() &&
	                   !image.IsBufferModified() && image.untracked_holes.empty() &&
	                   !m_buffer_cache.HasGpuDirtyBytes(info.data.address, info.data.size) &&
	                   HashGuestRange(info.data.address, info.data.size) == hash;
	if (igual) {
		FinishRefresh(image);
		return id;
	}
	FreeImage(id, "tex-dedup-verifica");
	Image::NoteDedupSeparated();
	return InsertImage(info, may_fail);
}

// (M7, GestorMem) Bajo m_lock (RunGarbageCollector, una vez por flip).
void TextureCache::GestorPass(uint64_t tick) {
	if (!GestorMem::On()) return;
	static bool registrado = false;
	if (!registrado) {
		registrado = true;
		GestorMem::RegistrarExpulsor(GestorMem::TipoTextura, GestorMem::Vram, &GestorTex::ExpulsorTexturas, nullptr);
		GestorMem::RegistrarExpulsor(GestorMem::TipoCopiaRam, GestorMem::Ram, &GestorTex::ExpulsorCopiasRam, nullptr);
		GestorTex::copias_id = GestorMem::Registrar(GestorMem::TipoCopiaRam, GestorMem::Ram, 0, GestorMem::FlagRespaldo, nullptr);
		RamGuard::RegisterFlush("gestor-mem: copias RAM de texturas", &GestorTex::FlushRamGuard);
		// La RAM que ram-guard retiene para simular un PC mas pequeno (KYTY_SIMULATE_RAM_MB): fija, solo para el cuadre.
		if (const uint64_t lastre = RamGuard::RamBallast(); lastre != 0)
			(void)GestorMem::Registrar(GestorMem::TipoOtro, GestorMem::Ram, lastre, GestorMem::FlagFijado, nullptr);
	}
	// 1) Candidatas que dejo el gestor (texturas con respaldo, sin uso en su edad minima).
	std::vector<GestorMem::Candidato> lista;
	{
		std::lock_guard lk(GestorTex::cola_m);
		lista.swap(GestorTex::cola);
	}
	for (const auto& c: lista) {
		if (!GestorMem::Vivo(c.id)) continue;
		const ImageId id    = GestorTex::DeDueno(c.dueno);
		auto*         image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered || image->gestor_id != c.id || image->IsSparse() || !GuestTexture(*image) ||
		    image->SharedWithOthers() || m_lru_cache.TickOf(image->lru_id) + GestorMem::InUseFrames > tick ||
		    (TexNativa::On() && TexNativa::AnyOwned(image->info.data.address, image->info.data.End())) ||
		    TextureOffload::Overlaps(image->info.data.address, image->info.data.size) ||
		    HasPendingDownload(image->info.data.address, image->info.data.size)) {
			GestorTex::c_saltadas.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		const uint64_t size = image->info.data.size;
		VramReparto::Evicted(image->info.data.address, size); // (re-subidas: GestorMem::Resubido)
		VramSaverAdd(VramEvicted, 1);
		VramSaverAdd(VramEvictedBytes, size);
		DeleteImage(id);
		GestorTex::c_n.fetch_add(1, std::memory_order_relaxed);
		GestorTex::c_bytes.fetch_add(size, std::memory_order_relaxed);
	}
	// 2) Copias RAM de texturas (cada ~5 s): bytes de la RAM del juego de texturas limpias que ya estan en la VRAM.
	if (tick % 300 == 0 && GestorTex::copias_id != 0) {
		uint64_t sum = 0;
		m_slot_images.ForEach([&](ImageId /*id*/, const Image& image) {
			if (GuestTexture(image) && !image.IsCpuDirty() && image.backing.image != nullptr && !image.IsSparse() &&
			    !TextureOffload::Overlaps(image.info.data.address, image.info.data.size))
				sum += image.info.data.size;
		});
		GestorMem::CambiarTamano(GestorTex::copias_id, sum);
	}
	// 3) Linea [tex-dedup] cada 5 s.
	const auto d = Image::Dedup();
	TexStream::c_dedup_n.store(d.ahora_n, std::memory_order_relaxed);
	TexStream::c_dedup_bytes.store(d.ahora_bytes, std::memory_order_relaxed);
	static auto     last = std::chrono::steady_clock::now();
	static uint64_t l_n = 0, l_b = 0, l_s = 0;
	if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(5)) {
		last              = now;
		const uint64_t n = GestorTex::c_n.load(), b = GestorTex::c_bytes.load(), s = GestorTex::c_saltadas.load();
		std::printf("[tex-dedup] 5 s: %llu imagenes comparten VkImage ahora = %llu MiB de VRAM ahorrados; unidas en total %llu "
		            "(%llu MiB), separadas por escritura (copy-on-write) %llu, escrituras en compartida sin separar %llu (debe "
		            "ser 0); indice %zu | [gestor-tex] texturas expulsadas por el gestor %llu (%llu MiB) en 5 s, saltadas %llu\n",
		            static_cast<unsigned long long>(d.ahora_n), static_cast<unsigned long long>(d.ahora_bytes >> 20u),
		            static_cast<unsigned long long>(d.unidas), static_cast<unsigned long long>(d.unidas_bytes >> 20u),
		            static_cast<unsigned long long>(d.separadas), static_cast<unsigned long long>(d.escrituras_tarde),
		            TexDedup::indice.size(), static_cast<unsigned long long>(n - l_n),
		            static_cast<unsigned long long>((b - l_b) >> 20u), static_cast<unsigned long long>(s - l_s));
		std::fflush(stdout);
		l_n = n, l_b = b, l_s = s;
	}
}

bool TextureCache::PartialDirtyCandidate(const Image& image) {
	const auto& info = image.info;
	// Surface metadata (DCC of a color target) does not change how guest data uploads: the
	// whole-image path uploads the same subresources from the same guest bytes. A depth array
	// in a texture tile mode (the game's streamed shadow maps) uploads as a texture too; one
	// that uploads as a depth target takes the whole-image path at refresh (UploadImagePartial).
	return kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) != 0 &&
	       info.data.size >= PartialDirtyCandidateSize && !image.depth_id &&
	       image.backing.image != nullptr && info.samples == 1 && !info.IsVolume() &&
	       !info.HasStencil() && info.metadata.compression == VideoOutCompression::Uncompressed;
}

// An image the GPU wrote stays GPU-owned outside the invalidated span: its other bytes are the GPU's, the span is
// uploaded from guest memory at the next use (RefreshImage -> InitializeImage takes the partial path; a depth array
// uploads whole layers, UploadDepthPartial; a target binding refreshes it before the GPU writes again), as on the
// console, where a CPU write or a remap replaces only those bytes. A whole-image invalidation uploaded everything from
// memory, which does not hold what the GPU rendered. The game's shadow-map pool (an 80-layer depth array, 320 MiB) lends
// its free layers to the texture streamer, which remaps one layer at a time and tiles textures into it from the CPU
// while shadows render into others: the whole array went to the GPU again every few seconds (100-270 ms frames) and the
// rendered layers were lost.
bool TextureCache::TryInvalidatePartial(Image& image, uint64_t address, uint64_t size,
                                        uint64_t granule) {
	const auto& info = image.info;
	// (With GPU buffer writes over known ranges of it too: a refresh uploads those with the dirty ranges, InitializeImage.
	// One written all over, or where not known, goes up whole anyway.)
	if (!PartialDirtyCandidate(image) || !image.registered || !image.IsTracked() ||
	    image.track_addr != info.data.address || image.track_addr_end != info.data.End() ||
	    !image.CanTakePartialDirty() || (image.IsBufferModified() && image.BufferModifiedRanges().empty()) ||
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
		WatchImagePages<false>(m_page_manager, image, begin, size);
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
		WatchImagePages<false>(m_page_manager, image, address, size);
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

void TextureCache::MarkImageGranules(uint64_t address, uint64_t size) const {
	if (size == 0 || address >= ImageGranuleSpace) return;
	if (m_image_granules.empty()) m_image_granules.assign((ImageGranuleSpace >> ImageGranuleBits) / 64, 0);
	const uint64_t last = (std::min(address + size, ImageGranuleSpace) - 1) >> ImageGranuleBits;
	for (uint64_t g = address >> ImageGranuleBits; g <= last; ++g) m_image_granules[g >> 6] |= uint64_t {1} << (g & 63);
}

bool TextureCache::MayHaveImages(uint64_t address, uint64_t size) const {
	if (m_image_granule_releases > 4096 && kyty_local_image_granules_mode.load(std::memory_order_relaxed) != 0) {
		// Released images leave their bits; rebuild from the registered ones.
		std::fill(m_image_granules.begin(), m_image_granules.end(), 0);
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered) MarkImageGranules(image.info.data.address, image.info.data.size);
		});
		m_image_granule_releases = 0;
	}
	return MayHaveImagesRead(address, size);
}

bool TextureCache::MayHaveImagesRead(uint64_t address, uint64_t size) const {
	if (kyty_local_image_granules_mode.load(std::memory_order_relaxed) == 0 || m_image_granule_releases > 4096 ||
	    size == 0 || address >= ImageGranuleSpace || size > ImageGranuleSpace - address)
		return true;
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

bool TextureCache::ViewFitsImage(const ImageDesc& desc) {
	const auto& view      = desc.view_info;
	const auto& resources = desc.info.resources;
	return view.level_count != 0 && view.layer_count != 0 && view.base_level < resources.levels &&
	       view.level_count <= resources.levels - view.base_level &&
	       (desc.info.IsVolume() ||
	        (view.base_layer < resources.layers && view.layer_count <= resources.layers - view.base_layer));
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || !ViewFitsImage(desc)) {
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
		NoteNativeGpuWrite(destination);
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
		NoteNativeGpuWrite(destination);
	}
}

// (Defined with UploadDepthPartial's helpers below.)
static void DepthArrayEvent(const char* what, const Image& image, uint64_t address, uint64_t size, const char* detail);

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

// KYTY_R32_SHADOW_POOL (on unless 0): the spot-shadow pool (a 360-layer 384x384 array at 0x3f...) is written by a
// compute pass as an R32 float storage image and sampled by the lighting with a depth compare (D32). Each switch
// recreated it (a 135 MiB copy each way, 7 pairs a session). Where the GPU compares depth on R32_SFLOAT
// (VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT, Vulkan 1.3; and linear filtering for PCF), the R32 image
// is sampled with the compare as it is: the pool stays R32 and is never switched back to D32 for sampling.
bool TextureCache::R32DepthCompareSupported() const {
	static const bool supported = [this] {
		const char* value = std::getenv("KYTY_R32_SHADOW_POOL");
		if (value != nullptr && value[0] == '0') return false;
		vk::FormatProperties3 properties3 {};
		vk::FormatProperties2 properties2 {};
		properties2.pNext = &properties3;
		m_graphics.physical_device.getFormatProperties2(vk::Format::eR32Sfloat, &properties2);
		const auto needed = vk::FormatFeatureFlagBits2::eSampledImageDepthComparison |
		                    vk::FormatFeatureFlagBits2::eSampledImageFilterLinear |
		                    vk::FormatFeatureFlagBits2::eSampledImage;
		const bool ok = (properties3.optimalTilingFeatures & needed) == needed;
		std::printf("Depth array: R32 float depth compare %s (KYTY_R32_SHADOW_POOL=0: off)\n",
		            ok ? "supported: the shadow pool stays R32" : "unsupported: the shadow pool switches to D32 to be sampled");
		std::fflush(stdout);
		return ok;
	}();
	return supported;
}

// KYTY_CUBE_SHADOW_FIX (on unless 0): a layered shadow pool the GPU rendered (the 384x384x360 D16 cube pool, written as
// R16 storage by a compute pass and then sampled with a depth compare again) switches depth <-> colour with a full copy
// both ways (bit-exact: D16 and R16 through a buffer). The image it leaves stayed parked one flip only: the next switch,
// seconds later in a burst of them, made a new 135 MiB image (and its views) each time. Now the one it leaves stays
// parked as its twin for CubeTwinKeep collector runs (taken back at the next switch, copied over whole as before;
// freed first when video memory runs short). Contents are handled as before: only the image's lifetime changes.
static bool CubeShadowFixOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_CUBE_SHADOW_FIX");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}
static constexpr uint64_t CubeTwinKeep = 1800;

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if (!cached.info.IsDepth() && !requested.IsDepth()) {
		return {};
	}
	// (Defined here: a depth-compare texture of an R32 float array of the same layout samples it as it is.)
	const auto R32ShadowPoolSample = [this](const ImageInfo& want, const Image& have) {
		return want.pixel_format == vk::Format::eD32Sfloat && have.backing.format == vk::Format::eR32Sfloat &&
		       have.info.pixel_format == vk::Format::eR32Sfloat && have.backing.samples == 1 && want.samples == 1 &&
		       want.data == have.info.data && want.extent == have.info.extent &&
		       want.resources == have.info.resources && want.resources.levels == 1 && want.resources.layers > 1 &&
		       want.type == have.info.type && want.pitch == have.info.pitch && want.tile_mode == have.info.tile_mode &&
		       !want.HasStencil() && !have.info.HasStencil() && !want.HasMetadata() && !have.info.HasMetadata() &&
		       R32DepthCompareSupported();
	};
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
			if (requested.IsDepth() && !cached.info.IsDepth() && R32ShadowPoolSample(requested, cached)) {
				return cached_id;
			}
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
	static const char* const binding_names[] = {"texture", "storage", "render_target", "depth_target", "video_out"};
	const auto binding_index = static_cast<uint32_t>(binding);
	char detail[256];
	std::snprintf(detail, sizeof(detail),
	              "binding=%s requested: depth=%d fmt=%u guest_fmt=%u layers=%u levels=%u type=%u tile=%u cached: fmt=%u levels=%u",
	              binding_index < 5 ? binding_names[binding_index] : "?", requested.IsDepth() ? 1 : 0,
	              static_cast<uint32_t>(requested.pixel_format), static_cast<uint32_t>(requested.guest_format),
	              requested.resources.layers, requested.resources.levels, static_cast<uint32_t>(requested.type),
	              static_cast<uint32_t>(requested.tile_mode), static_cast<uint32_t>(cached.info.pixel_format),
	              cached.info.resources.levels);
	DepthArrayEvent(binding == BindingType::DepthTarget ? "a recreation for a depth target" : "a recreation for another use", cached,
	                requested.data.address, requested.data.size, detail);
	// The spot-shadow pool (a GPU-rendered layered depth image) taken over by another use keeps its guard: the texture
	// streamer's CPU writes to its free layers then refresh only those layers, never all of it from guest memory,
	// which never held the rendered shadows (they vanished until the lights were rendered again).
	const bool alias_guard = (cached.info.IsDepth() || cached.depth_array_alias) && cached.IsGpuModified() &&
	                         cached.info.resources.layers > 1;
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
	info.htile_clear_mask = 0;
	auto replacement_id   = TakeParkedImage(info);
	const bool from_parked = static_cast<bool>(replacement_id);
	if (!replacement_id) replacement_id = InsertImage(info);
	const auto copy_start = std::chrono::steady_clock::now();
	auto& replacement = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	replacement.depth_array_alias = alias_guard && !replacement.info.IsDepth() &&
	                                replacement.info.resources.layers == cached.info.resources.layers;
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
		RtNativo::Hereda(cached, replacement); // (RT-NATIVO) sus paginas no estan vigiladas: la sustituta tampoco
		replacement.untracked_holes = std::move(cached.untracked_holes);
		cached.untracked_holes.clear();
		cached.track_addr     = 0;
		cached.track_addr_end = 0;
	}
	if (alias_guard) {
		// KYTY_DEPTH_ARRAY_LOG: what the switch did (the copy keeps the rendered layers; a replacement that is not
		// GPU-modified afterwards, or a skipped copy, loses them).
		static std::atomic<uint32_t> switch_reports {0};
		if (switch_reports.fetch_add(1, std::memory_order_relaxed) < 64) {
			std::printf("Depth array switch: %s -> %s replacement=%s copied_samples=%u/%u gpu_after=%d cpu_after=%d "
			            "partial_after=%d buffer_after=%d copy_ms=%.2f frame=%llu command_serial=%llu tick=%llu speculative=%d "
			            "packet_slot=%zu packet=%llu\n",
			            cached.info.IsDepth() ? "depth" : "color", replacement.info.IsDepth() ? "depth" : "color",
			            from_parked ? "parked" : "new", cached.backing.samples, replacement.backing.samples,
			            replacement.IsGpuModified() ? 1 : 0, replacement.IsCpuDirty() ? 1 : 0,
			            replacement.IsPartiallyCpuDirty() ? 1 : 0, replacement.IsBufferModified() ? 1 : 0,
			            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copy_start).count(),
			            static_cast<unsigned long long>(m_frame.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(m_scheduler.CommandSerial()),
			            static_cast<unsigned long long>(m_scheduler.CurrentTick()), Spec::t_effects != nullptr ? 1 : 0,
			            Spec::t_packet_slot, static_cast<unsigned long long>(Spec::t_packets));
			std::fflush(stdout);
		}
	}
	const bool keep_twin = alias_guard && CubeShadowFixOn() && cached.info.resources.levels == 1 &&
	                       replacement.info.resources.layers == cached.info.resources.layers &&
	                       replacement.info.data == cached.info.data && cached.backing.samples == 1;
	if (alias_guard) {
		ShadowPoolTrace::Count(ShadowPoolTrace::Find(cached.info.data.address, cached.info.resources.layers,
		                                             cached.info.extent.width, cached.info.extent.height),
		                       &ShadowPoolTrace::Pool::switches);
		static std::atomic<bool> told {false};
		if (keep_twin && !told.exchange(true)) {
			std::printf("Cube shadow pool: 0x%llx+0x%llx (%u layers %ux%u) switches %s <-> %s; the image it leaves stays "
			            "parked as its twin for %llu flips (KYTY_CUBE_SHADOW_FIX=0: one flip, then a new image)\n",
			            static_cast<unsigned long long>(cached.info.data.address),
			            static_cast<unsigned long long>(cached.info.data.size), cached.info.resources.layers,
			            cached.info.extent.width, cached.info.extent.height,
			            cached.info.IsDepth() ? "depth" : "color", replacement.info.IsDepth() ? "depth" : "color",
			            static_cast<unsigned long long>(CubeTwinKeep));
			std::fflush(stdout);
		}
	}
	ParkImage(cached_id, keep_twin ? CubeTwinKeep : 1);
	return replacement_id;
}

// (FreeImage's, but parked: TakeParkedImage takes it back for its description; the collector retires it a flip on.)
void TextureCache::ParkImage(ImageId id, uint64_t keep) {
	// (11-25d, revision 4) Con bytes "solo en la GPU": a la RAM antes de aparcar (aparcada, el recolector la retira sin
	// mirar, y una imagen nueva en su memoria los daria por ya subidos). Si no se puede descargar aqui, no se aparca: fuera
	// del registro y a NativeRestoreLater (sus bytes a la RAM en el siguiente drenaje, luego retirada).
	if (auto* native = m_slot_images.try_get(id); native != nullptr && native->registered && TexNativa::On() &&
	                                              TexNativa::AnyOwned(native->info.data.address, native->info.data.End())) {
		if (!NativeRestoreBeforeDelete(*native, true)) {
			if (DetachImage(id)) {
				NativeRestoreLater().emplace_back(id, TexNativa::EpocaActual());
				TexNativa::c_later_n.fetch_add(1, std::memory_order_relaxed);
				g_native_restore_later.store(true, std::memory_order_release);
			}
			return;
		}
	}
	auto& image = m_slot_images[id];
	if (image.IsGpuModified()) {
		image.ClearGpuModified();
	}
	image.depth_array_alias = false;
	if (image.share != nullptr) { // (M7, TEX-DEDUP) una VkImage compartida no se reutiliza para otra descripcion
		if (DetachImage(id)) RetireImage(id);
		return;
	}
	const auto address = image.info.data.address;
	if (!DetachImage(id)) return;
	auto& parked = m_parked[address];
	if (parked.id) RetireImage(parked.id);
	parked = {id, m_gc_tick, std::max<uint64_t>(keep, 1)};
}

// An image a depth overlap parked at `info`'s memory made for the same Vulkan image (all of its description but the
// metadata, which ImageInfo leaves out of the Vulkan image): as InsertImage's new one, registered again.
ImageId TextureCache::TakeParkedImage(const ImageInfo& info) {
	const auto parked = m_parked.find(info.data.address);
	if (parked == m_parked.end()) return {};
	auto*       image = m_slot_images.try_get(parked->second.id);
	const auto& held  = image != nullptr ? image->info : info;
	if (image == nullptr || image->registered || !(held.data == info.data) || !(held.stencil == info.stencil) ||
	    held.pixel_format != info.pixel_format || held.guest_format != info.guest_format || held.type != info.type ||
	    held.extent != info.extent || held.resources != info.resources || held.pitch != info.pitch ||
	    held.bytes_per_block != info.bytes_per_block || held.samples != info.samples || held.tile_mode != info.tile_mode ||
	    held.bgra16 != info.bgra16 || held.mip_layout != info.mip_layout || info.data.Empty())
		return {};
	const auto id = parked->second.id;
	m_parked.erase(parked);
	image->PrepareReuse(info);
	RegisterImage(id);
	return id;
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
	// KYTY_OVERLAP_WRITEBACK (on unless 0; 1-2 byte planes only): a compute-written image partly under the requested one, used in the last
	// frames, is written back to guest memory and freed now instead of living on beside it. The archstone menu's
	// preview video is decoded on the GPU into Y/UV planes whose addresses shift by 0x200000 from one video to the
	// next: the old planes stayed (gpu-modified, not in memory) and the new video's first reference frames came out of
	// memory without them, garbled until the next key frame. Not for render targets (two targets the game renders into
	// the same transient memory each frame would delete each other, see NumFramesBeforeRemoval).
	const auto retire_written = [&]() {
		// 1 (default): written back, then freed; 2: freed without writing back (the new image starts from what the
		// game left in memory, e.g. a clear it made itself); 0: off.
		static const int mode = [] {
			const char* value = std::getenv("KYTY_OVERLAP_WRITEBACK");
			return value == nullptr ? 1 : std::atoi(value);
		}();
		const bool enabled = mode != 0;
		// Only narrow planes (1-2 bytes a pixel: video Y/UV/alpha) of the same tiling: without this it fired on two
		// RGBA16F compute images the game swaps in the same memory every frame (45-57 ms frames, 13-10).
		if (cached.info.bytes_per_block > 2 || requested.bytes_per_block > 2 ||
		    cached.info.tile_mode != requested.tile_mode)
			return false;
		if (!enabled || safe_to_delete || !cached.IsGpuModified() || !cached.usage.storage ||
		    cached.usage.render_target || cached.usage.depth_target || cached.binding.is_target || cached.depth_id)
			return false;
		if (mode != 2 && (!SafeToDownload(cached) || !TryDownloadImage(cached_id))) return false;
		cached.ClearGpuModified();
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 200)
			std::printf("OVERLAP-WRITEBACK mode=%d cached=0x%llx+0x%llx requested=0x%llx+0x%llx\n",
			            mode, static_cast<unsigned long long>(cached.info.data.address),
			            static_cast<unsigned long long>(cached.info.data.size),
			            static_cast<unsigned long long>(requested.data.address),
			            static_cast<unsigned long long>(requested.data.size));
		FreeImage(cached_id, "overlap-gpu-written");
		return true;
	};

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
		if (const auto texel_id = ResolveBlockTexelOverlap(requested, cached_id)) {
			return {texel_id};
		}
		// (KYTY_BLOCK_CLASS_OVERLAP_REPLACE, upstream 495e2a8: of the other block class and not its texels or blocks, the
		// cached image does not hold the requested one's bytes: the checks below would return it as a view of another format.)
		if (requested.IsBlock() != cached.info.IsBlock() && BlockClassOverlapReplaceOn()) {
			// (A compressed request over texels: ExpandImage copies the levels both hold, whichever has more.)
			const char* why = requested.IsBlock() ? BlockTexelRefusal(cached.info, requested, true)
			                                      : BlockTexelRefusal(requested, cached.info);
			// (Both only sampled: an image of its own beside the other; a write binding at the address frees the other.)
			if (why != nullptr && GuestTextureRequest(binding, requested) && GuestTexture(cached)) {
				NoteGuestAlias(requested, cached);
				return {merged_id};
			}
			if (why != nullptr) {
				ReplaceBlockClassOverlap(requested, cached_id, why);
				return {merged_id};
			}
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
			// (A sampled view of a guest texture in a shape it does not hold, and the expanded image would not hold the
			// texture's own: an image of its own, GuestTexture. One holding it replaces it as before.)
			if (GuestTextureRequest(binding, requested) && GuestTexture(cached) &&
			    !requested.resources.Contains(cached.info.resources)) {
				NoteGuestAlias(requested, cached);
				return {merged_id};
			}
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
		if (retire_written()) return {};
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
	if (mip < 0 || cached.info.SliceOf(requested, mip) < 0) {
		// (The requested image starts before the cached one and does not hold it as a level or layer.)
		if (retire_written()) return {merged_id};
	}
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (layer >= 0) {
			if (cached.binding.is_target) {
				cached.binding.needs_rebind = true;
				if (merged_id) {
					auto& merged = m_slot_images[merged_id];
					if (!merged.binding.is_target) ++merged.validity;
					merged.binding.is_target = true;
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

// The blocks of a block-compressed image as an uncompressed image of them: the same tiling, element grid (BlockExtent,
// bytes per block: ResolveOverlap's checks before) and placement of the levels both hold. The game's GPU block
// compressors write BC textures so (Demon's Souls' character creation: each head preset's face, composited and then
// compressed to BC3 / BC7 2048x2048 by CS 55d35da67c9a387a / b59223a986bb112e through R32G32B32A32_UINT 512x512 storage
// views of the textures' bytes). Vulkan views of one image cannot be that over more than a level, so the requested form
// is an image of its own made from the other's contents, and the other is parked (ExpandImage takes it back the other
// way, with the texels' contents). Before, the storage view was an image of its own made from the guest's bytes and the
// compressed texture kept its old contents: faces showed the previous preset's makeup, tattoos or stale bytes.
// (upstream 495e2a8, PR #374) Why the uncompressed `texels` cannot be the texels of the block-compressed `blocks`.
const char* TextureCache::BlockTexelRefusal(const ImageInfo& texels, const ImageInfo& blocks, bool common_levels) {
	if (texels.IsBlock() || !blocks.IsBlock()) return "block class";
	if (texels.IsDepth() || blocks.IsDepth() || texels.samples != 1 || blocks.samples != 1) return "depth or samples";
	if (texels.tile_mode != blocks.tile_mode) return "tiling";
	if (texels.type != blocks.type || texels.IsVolume()) return "type";
	if (texels.resources.layers != blocks.resources.layers) return "layers";
	if (texels.HasMetadata() || blocks.HasMetadata() || texels.HasStencil() || blocks.HasStencil())
		return "metadata or stencil";
	const uint32_t levels =
	    common_levels ? std::min(texels.resources.levels, blocks.resources.levels) : texels.resources.levels;
	if ((!common_levels && texels.resources.levels > blocks.resources.levels) || levels > texels.mip_layout.size() ||
	    levels > blocks.mip_layout.size())
		return "levels";
	for (uint32_t level = 0; level < levels; ++level) {
		if (texels.mip_layout[level].offset != blocks.mip_layout[level].offset ||
		    texels.mip_layout[level].size != blocks.mip_layout[level].size)
			return "level placement";
	}
	return nullptr;
}

ImageId TextureCache::ResolveBlockTexelOverlap(const ImageInfo& requested, ImageId cached_id) {
	if (BlockTexelRefusal(requested, m_slot_images[cached_id].info) != nullptr) return {};
	RefreshCopySource(cached_id);
	auto texel_id = TakeParkedImage(requested);
	if (!texel_id) texel_id = InsertImage(requested);
	auto& texels = m_slot_images[texel_id];
	auto& blocks = m_slot_images[cached_id];
	texels.usage = blocks.usage;
	if (blocks.binding.is_bound || blocks.binding.is_target) {
		blocks.binding.needs_rebind = true;
	}
	InitializeImage(texel_id);
	CopyImage(texel_id, cached_id);
	ParkImage(cached_id);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::printf("TextureCache: 0x%llx+0x%llx as the texels of its blocks (fmt %u %ux%u levels %u from fmt %u %ux%u "
		            "levels %u)\n",
		            static_cast<unsigned long long>(requested.data.address),
		            static_cast<unsigned long long>(requested.data.size), static_cast<uint32_t>(requested.guest_format),
		            requested.extent.width, requested.extent.height, requested.resources.levels,
		            static_cast<uint32_t>(m_slot_images[cached_id].info.guest_format),
		            m_slot_images[cached_id].info.extent.width, m_slot_images[cached_id].info.extent.height,
		            m_slot_images[cached_id].info.resources.levels);
	}
	return texel_id;
}

// KYTY_BLOCK_CLASS_OVERLAP_REPLACE (upstream 495e2a8, PR #374; default 1; 0 off, read once): an image registered at
// the requested address of the other block class (block-compressed against uncompressed) that is not its texels or
// blocks is replaced: parked, and the requested image made from guest memory as any new one. Before, ResolveOverlap's
// later checks returned it whenever the formats shared a size class, and the binding viewed that image's bytes in the
// other format. Demon's Souls: after some reloads a 2048x2048 BC1 texture of the previous load stayed registered where
// the water simulation's 512x512 R32G32_UINT state now lives: a spiked stream-bed sheet after Continue.
bool TextureCache::BlockClassOverlapReplaceOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_BLOCK_CLASS_OVERLAP_REPLACE");
		const bool  set   = value != nullptr && value[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("Texture cache: an image of the other block class at a requested address replaced, not viewed: %s "
		            "(KYTY_BLOCK_CLASS_OVERLAP_REPLACE=%s)\n",
		            set ? "on" : "off", set ? "0 turns it off" : "0");
		std::fflush(stdout);
		return set;
	}();
	return on;
}

void TextureCache::ReplaceBlockClassOverlap(const ImageInfo& requested, ImageId cached_id, const char* why) {
	// (A speculation's recorder thread changes no registration: the GPU thread translates the binding again.)
	if (CommandScheduler::ThreadRecorder() != nullptr)
		throw CommandScheduler::RecorderRefusal {"image of the other block class"};
	auto&      cached      = m_slot_images[cached_id];
	const bool gpu_written = cached.IsGpuModified();
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (++m_block_class_replaced <= 32) {
		std::printf("Texture cache: 0x%llx: fmt %u %ux%u levels %u (gpu-written %d) replaced for fmt %u %ux%u levels %u "
		            "(%s, refused for: %s); %llu replaced so far\n",
		            static_cast<unsigned long long>(requested.data.address), static_cast<uint32_t>(cached.info.guest_format),
		            cached.info.extent.width, cached.info.extent.height, cached.info.resources.levels, gpu_written ? 1 : 0,
		            static_cast<uint32_t>(requested.guest_format), requested.extent.width, requested.extent.height,
		            requested.resources.levels,
		            requested.IsBlock() ? "the cached image is not the texels of the requested blocks"
		                                : "the request is not the texels of the cached blocks",
		            why,
		            static_cast<unsigned long long>(m_block_class_replaced));
		std::fflush(stdout);
		m_block_class_logged = m_block_class_replaced;
	}
	ParkImage(cached_id);
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	// (The blocks' image parked for these texels: ResolveBlockTexelOverlap.)
	auto expanded_id = TakeParkedImage(info);
	if (!expanded_id) expanded_id = InsertImage(info);
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
			                        plan.LinearSize(), plan.tiles, source.Usage() == MemoryUsage::Upload);
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
		    m_tiler.Detile(source.Handle(), source_offset, info.data.size, info.data.size, tiles,
		                   source.Usage() == MemoryUsage::Upload);
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
	    m_tiler.Detile(source.Handle(), source_offset, info.stencil.size, info.stencil.size, tiles,
	                   source.Usage() == MemoryUsage::Upload);
	for (auto& copy: copies) {
		copy.bufferOffset += linear.offset;
	}
	image.Upload(copies, linear.buffer, linear.offset, linear.size);
}

// Local diagnostic: why the latest UploadImagePartial fell back to a whole upload (SLOW log).
static thread_local const char* g_partial_fail = "";

// KYTY_UPLOAD_LOG: a partial refresh's staged bytes (UPLOAD lines are the whole ones).
static void LogPartialUpload(const Image& image, size_t pieces, uint64_t bytes) {
	static const bool log_uploads = std::getenv("KYTY_UPLOAD_LOG") != nullptr;
	if (!log_uploads) return;
	std::printf("[tsc %llu] UPLOAD-PART addr=0x%llx size=0x%llx layers=%u levels=%u pieces=%zu bytes=0x%llx serial=%llu\n",
	            static_cast<unsigned long long>(__rdtsc()), static_cast<unsigned long long>(image.info.data.address),
	            static_cast<unsigned long long>(image.info.data.size), image.info.resources.layers,
	            image.info.resources.levels, pieces, static_cast<unsigned long long>(bytes),
	            static_cast<unsigned long long>(image.serial));
}
static bool PartialFail(const char* why) {
	g_partial_fail = why;
	return false;
}

// KYTY_DEPTH_ARRAY_GUARD (on unless 0): a layered depth image the GPU rendered into (the game's 80-layer shadow-map
// pool, 320 MiB, whose free layers its texture streamer borrows from the CPU) refreshes only its dirty layers, however
// many, never all of it from guest memory (see UploadDepthPartial). KYTY_DEPTH_ARRAY_LOG=1: every event that still
// drops such an image's rendered layers (a whole upload, a whole CPU-dirty state, a recreation, an unmap), not only
// the first 16.
static bool DepthArrayGuardOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DEPTH_ARRAY_GUARD");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}
static bool DepthArrayGuarded(const Image& image) {
	return DepthArrayGuardOn() && image.IsGpuModified() && (image.info.IsDepth() || image.depth_array_alias) &&
	       image.info.resources.layers > 1;
}
static void DepthArrayEvent(const char* what, const Image& image, uint64_t address, uint64_t size, const char* detail) {
	if (!(image.info.IsDepth() || image.depth_array_alias) || image.info.resources.layers < 2 || !image.IsGpuModified()) return;
	static const bool all = [] {
		const char* value = std::getenv("KYTY_DEPTH_ARRAY_LOG");
		return value != nullptr && value[0] == '1';
	}();
	static std::atomic<uint32_t> reports {0};
	if (std::strncmp(what, "a recreation", 12) != 0) {
		ShadowPoolTrace::Count(ShadowPoolTrace::Find(image.info.data.address, image.info.resources.layers,
		                                             image.info.extent.width, image.info.extent.height),
		                       &ShadowPoolTrace::Pool::whole_uploads);
	}
	if (!all && reports.fetch_add(1, std::memory_order_relaxed) >= 16) return;
	std::printf("Depth array: %s (its rendered layers at risk): image 0x%llx+0x%llx (%u layers, %ux%u) at 0x%llx+0x%llx "
	            "cpu=%d partial=%d maybe=%d buffer=%d buffer_ranges=%zu tracked_whole=%d %s (KYTY_DEPTH_ARRAY_LOG=1: all)\n",
	            what, static_cast<unsigned long long>(image.info.data.address),
	            static_cast<unsigned long long>(image.info.data.size), image.info.resources.layers, image.info.extent.width,
	            image.info.extent.height, static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
	            image.IsCpuDirty() ? 1 : 0, image.IsPartiallyCpuDirty() ? 1 : 0, image.IsMaybeCpuDirty() ? 1 : 0,
	            image.IsBufferModified() ? 1 : 0, image.BufferModifiedRanges().size(),
	            image.track_addr == image.info.data.address && image.track_addr_end == image.info.data.End() ? 1 : 0,
	            detail != nullptr ? detail : "");
	std::fflush(stdout);
}

// A texture the game writes while draws sample it (mip streaming: its threads write the levels while the frame is
// translated) was dirty again at each draw, and each uploaded all of it: 6-14 times in a millisecond for one 2.75 MiB
// texture, 1.5 GB of the 2.85 GB uploaded in a 10 s walk. While the command buffer that recorded its last whole upload
// is open, those copies have not run: the upload worker copies the guest bytes again into the same staging, and
// that upload carries them. Images only sampled (the GPU never writes them): draws recorded since its upload read
// the bytes written up to the submission, as the console's GPU, a frame behind, reads them when it runs the draws.
// KYTY_UPLOAD_LOG: why the last RefillUpload did not refill.
static thread_local const char* g_refill_refusal = "";

bool TextureCache::RefillUpload(Image& image) {
	const auto refuse = [](const char* why) {
		g_refill_refusal = why;
		return false;
	};
	if (image.staged_ring == nullptr) return refuse("none");
	if (image.staged_serial != m_scheduler.CommandSerial() || m_scheduler.CurrentTick() >= CommandScheduler::PendingTick ||
	    m_scheduler.Current().IsInvalid())
		return refuse("submitted");
	if (image.IsBufferModified() || image.IsGpuModified() || image.usage.storage || image.usage.render_target ||
	    image.usage.depth_target || image.binding.is_target || UploadBinding(image) == BindingType::DepthTarget)
		return refuse("written");
	if (!m_buffer_cache.RefillImageStaging(*image.staged_ring, image.staged_offset, image.info.data.address,
	                                       image.info.data.size))
		return refuse("source");
	LiveCounters::Add(LiveCounters::ImageRefills);
	LiveCounters::Add(LiveCounters::ImageRefillBytes, image.info.data.size);
	return true;
}

void TextureCache::InitializeImage(ImageId id) {
	using Clock = std::chrono::steady_clock;
	LiveCounters::ScopedUs timed(LiveCounters::ImageInitUs);
	auto&             image = m_slot_images[id];
	NativeGuardDirty(image); // (N2+N3)
	Clock::time_point marks[3] {Clock::now()};
	uint64_t          hole_bytes_before = 0; // (picos1: what the track phase re-watched)
	const int         tracked_before    = image.IsTracked() ? 1 : 0;
	for (const auto& [hb, he]: image.untracked_holes) hole_bytes_before += he - hb;
	// (buffer: GPU buffer writes over known ranges; buffer-all: over all of it or where not known; +cpu: and CPU writes)
	const bool        buffer_all = image.IsBufferModified() && image.BufferModifiedRanges().empty();
	const bool        cpu        = image.IsCpuDirty();
	const char*       kind = image.IsPartiallyCpuDirty() ? (image.IsBufferModified() ? "partial+buffer" : "partial")
	                         : !image.IsBufferModified() ? (cpu ? "cpu" : "clean")
	                         : buffer_all                ? (cpu ? "buffer-all+cpu" : "buffer-all")
	                                                     : (cpu ? "buffer+cpu" : "buffer");
	SlowLog::Scope    slow([&](double ms) {
        const auto at = [&](int i) {
            return marks[i] == Clock::time_point {}
		               ? 0.0
		               : std::chrono::duration<double, std::milli>(marks[i] - marks[0]).count();
        };
        std::printf("SLOW InitializeImage %.1f ms addr=0x%llx size=0x%llx levels=%u layers=%u kind=%s track=%.1f "
		               "source=%.1f partial_fail=%s tracked_before=%d holes_mib=%.1f\n",
		               ms, static_cast<unsigned long long>(image.info.data.address),
		               static_cast<unsigned long long>(image.info.data.size), image.info.resources.levels,
		               image.info.resources.layers, kind, at(1), at(2), g_partial_fail, tracked_before,
		               static_cast<double>(hole_bytes_before) / (1024.0 * 1024.0));
    }, SlowLog::HitchThreshold());
	if (image.info.data.Empty()) {
		return;
	}
	// Guest bytes it reads that were dropped from RAM come back first (texture-offload.h), the asynchronous restores of
	// images gone too (submitted first: the offload worker writes them back when the GPU is done).
	if (TextureOffload::Overlaps(image.info.data.address, image.info.data.size)) {
		RestoreOffloaded(image.info.data.address, image.info.data.size, static_cast<uint8_t>(TextureOffload::Touch::Upload));
	}
	if (TextureOffload::IsRestoring(image.info.data.address, image.info.data.size)) {
		if (CommandScheduler::ThreadRecorder() != nullptr) throw CommandScheduler::RecorderRefusal {"texture offload restore"};
		if (m_offload_unsubmitted) {
			m_offload_unsubmitted = false;
			m_scheduler.Flush();
		}
		TextureOffload::WaitRestores(image.info.data.address, image.info.data.size);
		// (ram15) A range the worker was dropping meanwhile: dropped now, its bytes back before the upload reads them.
		if (TextureOffload::Overlaps(image.info.data.address, image.info.data.size))
			RestoreOffloaded(image.info.data.address, image.info.data.size, static_cast<uint8_t>(TextureOffload::Touch::Upload));
	}
	DeferredImageProtects protects {m_buffer_cache, m_page_manager};
	protects.Track([&] { TrackImage(id); });
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
	if (upload && RtNativo::Es(image)) { // (RT-NATIVO) se mide: solo su primer uso o tras una escritura de la GPU por buffer
		// (RT-NATIVO2) primera subida de RAM nunca escrita: clear a 0 en la GPU (mismo contenido, sin leer la RAM)
		if (RtNativo::Fase2() && !image.IsBufferModified() && image.IsCpuDirty() && !image.info.IsDepth() &&
		    !image.depth_id && !image.info.IsVolume() && image.backing.image != nullptr &&
		    RtNativo::RamVirgen(image.info.data.address, image.info.data.size)) {
			m_scheduler.EndRendering();
			const auto           command = m_scheduler.Current().Handle();
			vk::ClearColorValue  zero {};
			image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
			const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0,
			                                       VK_REMAINING_ARRAY_LAYERS};
			command.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &zero, 1, &range);
			image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
			              {}, command);
			image.staged_ring = nullptr;
			RtNativo::c_clears.fetch_add(1, std::memory_order_relaxed);
			RtNativo::c_clears_bytes.fetch_add(image.info.data.size, std::memory_order_relaxed);
			image.ClearBufferModified();
			FinishRefresh(image);
			return;
		}
		(image.IsBufferModified() ? RtNativo::c_sub_buffer : RtNativo::c_sub_cpu).fetch_add(1, std::memory_order_relaxed);
		RtNativo::c_subidas.fetch_add(1, std::memory_order_relaxed);
		RtNativo::c_subidas_bytes.fetch_add(image.info.data.size, std::memory_order_relaxed);
	}
	if (upload) {
		// A refill first: no GPU work at all, where a partial upload records some (and would end the refills: its
		// copies come after the whole upload's, which a refill would then overwrite with older bytes).
		bool uploaded = RefillUpload(image);
		if (uploaded && kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 2 &&
		    PartialDirtyCandidate(image)) {
			UpdatePartialHashes(image, true);
		}
		if (!uploaded && image.IsPartiallyCpuDirty() && !image.IsBufferModified()) {
			g_partial_fail = "";
			uploaded = UploadImagePartial(image);
			if (!uploaded) {
				LiveCounters::Add(LiveCounters::PartialFallbacks);
			} else {
				image.staged_ring = nullptr;
			}
		}
		// GPU buffer writes that only filled all of it with one 32-bit value since its last upload (compute fills over
		// it, TryConsumeComputeImageClear): a clear to that value, as an upload of those bytes would leave it. The game
		// clears render targets that way in 4 MiB pieces: at Stonefang every frame four 720p RGBA16F, a 1440p R32 array
		// of 4 and a 1440p target - 105 MiB of detiles of zeros a frame (10-14).
		if (!uploaded && image.IsBufferModified() && !image.IsCpuDirty() && !image.info.IsDepth() && !image.depth_id &&
		    !image.info.IsVolume() && image.backing.image != nullptr &&
		    kyty_local_fill_clear_mode.load(std::memory_order_relaxed) != 0) {
			vk::ClearValue clear {};
			if (const auto value = image.BufferFilledWith();
			    value.has_value() && DecodeFilledColorClear(image.info.pixel_format, *value, clear.color)) {
				m_scheduler.EndRendering();
				const auto command = m_scheduler.Current().Handle();
				image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
				const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0,
				                                       VK_REMAINING_ARRAY_LAYERS};
				command.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &clear.color, 1, &range);
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
				uploaded          = true;
				image.staged_ring = nullptr;
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::printf("TextureCache: image filled by compute fills cleared to 0x%08x instead of uploaded: addr=0x%llx "
					            "size=0x%llx fmt=%u %ux%u layers=%u\n",
					            *value, static_cast<unsigned long long>(image.info.data.address),
					            static_cast<unsigned long long>(image.info.data.size),
					            static_cast<uint32_t>(image.info.guest_format), image.info.extent.width,
					            image.info.extent.height, image.info.resources.layers);
				}
			}
		}
		// GPU buffer writes over known parts of it and no CPU write: the subresources over those parts alone, from the
		// buffers holding them. At Boletaria every frame a compute shader writes two 15 MiB buffers in memory shared with
		// render targets - a 1440p R32 array, 720p RGBA16F ones - and each of them was uploaded whole again: 105 MiB of
		// detiles a frame for 30 MiB written (10-13).
		if (!uploaded && image.IsBufferModified() && !image.IsCpuDirty() && !image.BufferModifiedRanges().empty() &&
		    kyty_local_buffer_ranges_mode.load(std::memory_order_relaxed) != 0) {
			const auto ranges = image.BufferModifiedRanges();
			uploaded          = UploadImageChunked(image, &ranges);
			if (uploaded) image.staged_ring = nullptr;
		}
		// CPU writes over known parts of it, and GPU buffer writes over known parts or none: the subresources over either
		// alone. The partial upload above takes no image with buffer writes and stages guest bytes alone: a texture pool's
		// range over memory the GPU writes too (issue #49: 1 GB of 1024x1024 slots) went up whole when the game streamed a
		// texture into it, 190-360 ms frames. The sources are the whole upload's (one for all of it below ChunkedUploadMin,
		// a piece at a time past it): asked for pieces, the buffer cache made and joined buffers inside the image's range
		// again and again (an 8 GB laptop's GPU faults in the culling chain came 3 times in 4 tours with that, 10-18).
		if (!uploaded && image.IsPartiallyCpuDirty() && !image.IsMaybeCpuDirty() &&
		    kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 1 &&
		    (!image.IsBufferModified() || (!image.BufferModifiedRanges().empty() &&
		                                   kyty_local_buffer_ranges_mode.load(std::memory_order_relaxed) != 0))) {
			auto ranges = image.CpuDirtyRanges();
			if (image.IsBufferModified()) {
				for (const auto& [begin, end]: image.BufferModifiedRanges()) Image::AddRange(ranges, begin, end);
			}
			uploaded = image.info.data.size >= ChunkedUploadMin ? UploadImageChunked(image, &ranges)
			                                                    : UploadImageSubresources(image, ranges);
			if (uploaded) image.staged_ring = nullptr;
		}
		if (!uploaded && UploadImageChunked(image)) {
			uploaded            = true;
			image.staged_ring   = nullptr;
			if (kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 2 &&
			    PartialDirtyCandidate(image)) {
				UpdatePartialHashes(image, true);
			}
		}
		if (!uploaded) {
			DepthArrayEvent("whole upload", image, image.info.data.address, image.info.data.size, kind);
			bool refillable = false;
			const auto [source, source_offset] =
			    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size, &refillable);
			marks[2] = Clock::now();
			if (source == nullptr) {
				EXIT("TextureCache: failed to obtain image upload source\n");
			}
			LiveCounters::Add(LiveCounters::FullUploads);
			LiveCounters::Add(LiveCounters::FullUploadBytes, image.info.data.size);
			static const bool log_uploads = std::getenv("KYTY_UPLOAD_LOG") != nullptr;
			if (log_uploads) {
				std::printf("[tsc %llu] UPLOAD %s addr=0x%llx size=0x%llx fmt=%u %ux%u layers=%u levels=%u tile=%u "
				            "target=%d usage=%d%d%d%d gpu=%d serial=%llu partial_fail=%s refill=%s\n",
				            static_cast<unsigned long long>(__rdtsc()), kind,
				            static_cast<unsigned long long>(image.info.data.address),
				            static_cast<unsigned long long>(image.info.data.size),
				            static_cast<uint32_t>(image.info.guest_format), image.info.extent.width,
				            image.info.extent.height, image.info.resources.layers, image.info.resources.levels,
				            static_cast<uint32_t>(image.info.tile_mode), image.binding.is_target ? 1 : 0,
				            image.usage.texture ? 1 : 0, image.usage.storage ? 1 : 0,
				            image.usage.render_target ? 1 : 0, image.usage.depth_target ? 1 : 0,
				            image.IsGpuModified() ? 1 : 0, static_cast<unsigned long long>(image.serial),
				            image.IsPartiallyCpuDirty() && !image.IsBufferModified() ? g_partial_fail : "",
				            g_refill_refusal);
			}
			UploadImage(image, *source, source_offset);
			image.staged_ring   = refillable ? source : nullptr;
			image.staged_offset = source_offset;
			image.staged_serial = m_scheduler.CommandSerial();
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
		// (rtq2) A mapping that covers the range does not mean every page of it is committed: build 7c84280-dirty crashed
		// in XXH3_hashLong_64b <- UploadDepthPartial on a 4 MiB depth array at 0x1036c00000 with only 0xbe000 committed
		// (the rest MEM_RESERVE). Read straight only when every page is readable now; otherwise a guarded copy, page by
		// page (a page that cannot be read hashes as zeros, as a range the guest never wrote), never a fault.
		uint64_t readable = 0;
		if (Libs::Graphics::HostMemoryQueryReadable(reinterpret_cast<uint64_t>(data), size, readable) && readable >= size) {
			return XXH3_64bits(data, static_cast<size_t>(size));
		}
		thread_local SoftRelease::Scratch guarded("texture hash guarded copy");
		const auto                        hold_g = guarded.Use();
		auto&                             out    = guarded.bytes;
		out.resize(size);
		constexpr uint64_t Page = 0x1000;
		for (uint64_t done = 0; done < size;) {
			const uint64_t at   = reinterpret_cast<uint64_t>(data) + done;
			const uint64_t step = std::min<uint64_t>(size - done, Page - (at & (Page - 1)));
			if (const uint64_t left = Common::GuardedCopy(out.data() + done, data + done, step); left != 0) {
				std::memset(out.data() + done + (step - left), 0, static_cast<size_t>(left));
			}
			done += step;
		}
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("rtq2: guest range 0x%llx size 0x%llx only 0x%llx readable at its start: hashed through a guarded "
			            "copy (no fault)\n",
			            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			            static_cast<unsigned long long>(readable));
			std::fflush(stdout);
		}
		return XXH3_64bits(out.data(), static_cast<size_t>(size));
	}
	// (A copy of the texture's guest bytes, up to hundreds of MiB, kept by the thread: KYTY_SOFT_RELEASE frees it idle.)
	thread_local SoftRelease::Scratch scratch("texture hash copy");
	const auto                        hold  = scratch.Use();
	auto&                             bytes = scratch.bytes;
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

// KYTY_SHADOW_REFRESH_FIX (on unless 0): a layer of a depth array the GPU rendered is refreshed from guest memory only
// when its guest bytes changed since the GPU took it. Guest memory never holds the GPU's depth: re-uploading bytes that
// did not change only replaces a rendered shadow with what the memory held before it. The game's 80-layer spot-shadow
// pool (1024x1024 D32 at 0x102a400000, its shadows in layers 0-12 kept from flip to flip) had such layers refreshed a
// few flips in ten at the Boletaria lamps (combo20 log 061301: 1-19 layers a flip, "Shadow pool" lines): the floor
// shadows under the lamps came and went with wrong shapes. A layer whose bytes did change (a CPU write, another
// mapping) is uploaded as before, and counts as the guest's until the GPU renders it again.
static bool ShadowRefreshFixOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_SHADOW_REFRESH_FIX");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}
enum : uint8_t { LayerRendered = 1, LayerDirtyByWrite = 2, LayerDirtyByMap = 4, LayerDirtyByUnmap = 8, LayerUploaded = 16 };
static bool LayerOwnerCandidate(const Image& image) {
	const auto& info = image.info;
	return ShadowRefreshFixOn() && info.IsDepth() && !info.IsVolume() && info.resources.levels == 1 &&
	       info.resources.layers >= 2 && info.samples == 1 && !info.data.Empty() &&
	       info.data.size % info.resources.layers == 0;
}

void TextureCache::NoteDepthLayersRendered(Image& image, uint32_t first, uint32_t count) {
	if (!LayerOwnerCandidate(image)) return;
	const uint32_t layers = image.info.resources.layers;
	if (first >= layers || count == 0) return;
	count       = std::min(count, layers - first);
	auto& owner = m_layer_owners[image.serial];
	if (owner.state.size() != layers) {
		owner.state.assign(layers, 0);
		owner.hash.assign(layers, 0);
	}
	const uint64_t slice = image.info.data.size / layers;
	for (uint32_t layer = first; layer < first + count; ++layer) {
		if ((owner.state[layer] & LayerRendered) != 0) continue;
		// (Its guest bytes as the GPU takes the layer: refreshed just before, RefreshDepthLayers.)
		owner.state[layer] = LayerRendered;
		owner.hash[layer]  = HashGuestRange(image.info.data.address + slice * layer, slice);
	}
}

void TextureCache::NoteDepthLayersDirtied(const Image& image, uint64_t begin, uint64_t end, uint8_t cause) {
	if (m_layer_owners.empty()) return;
	const auto found = m_layer_owners.find(image.serial);
	if (found == m_layer_owners.end()) return;
	auto&          owner  = found->second;
	const auto&    info   = image.info;
	const uint32_t layers = info.resources.layers;
	if (owner.state.size() != layers || layers == 0) return;
	const uint64_t slice = info.data.size / layers;
	begin                = std::max(begin, info.data.address);
	end                  = std::min(end, info.data.End());
	if (slice == 0 || begin >= end) return;
	const uint64_t first = (begin - info.data.address) / slice;
	const uint64_t last  = std::min<uint64_t>(layers, (end - info.data.address + slice - 1) / slice);
	for (uint64_t layer = first; layer < last; ++layer) {
		if ((owner.state[layer] & LayerRendered) != 0) owner.state[layer] |= cause;
	}
}

// texture streamer fills from the CPU): only those layers are staged, detiled and copied (of [first_layer, last_layer)
// only, for RefreshDepthLayers). The whole-image path staged and detiled every layer (320 MiB), 20-65 ms per refresh.
bool TextureCache::UploadDepthPartial(Image& image, uint32_t first_layer, uint32_t last_layer) {
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
	for (uint32_t layer = first_layer; layer < std::min(layers, last_layer); ++layer) {
		if (IntersectsRanges(dirty, base + slice * layer, base + slice * (layer + 1))) {
			picked.push_back(layer);
		}
	}
	// (KYTY_SHADOW_REFRESH_FIX: the rendered layers whose guest bytes did not change since the GPU took them stay.)
	// (Also a layer the GPU did not render through a depth bind of this image - the 80-layer pool's spot shadows had
	// 1-33 such layers re-uploaded a flip with "rendered layers kept 0", 20261027-182152 log, flickering lamp shadows:
	// once uploaded, its guest bytes are hashed, and a later refresh of the same bytes keeps the image's layer. Bytes
	// that did not change since the last upload can only replace the layer with what it already held or with less.)
	if (DepthArrayGuarded(image) && LayerOwnerCandidate(image) && !picked.empty()) {
		auto& slot = m_layer_owners[image.serial];
		if (slot.state.size() != layers) {
			slot.state.assign(layers, 0);
			slot.hash.assign(layers, 0);
		}
	}
	if (!m_layer_owners.empty() && DepthArrayGuarded(image)) {
		const auto found = m_layer_owners.find(image.serial);
		if (found != m_layer_owners.end() && found->second.state.size() == layers) {
			auto&    owner   = found->second;
			uint32_t kept    = 0;
			uint32_t changed = 0;
			uint32_t first_up = 0, reup = 0;
			std::erase_if(picked, [&](uint32_t layer) {
				auto& state = owner.state[layer];
				if ((state & LayerRendered) == 0) {
					const uint64_t begin = base + slice * layer;
					m_buffer_cache.DrainGuestReadback(begin, slice);
					const uint64_t now = HashGuestRange(begin, slice);
					if ((state & LayerUploaded) != 0 && owner.hash[layer] == now) {
						++kept;
						return true;
					}
					++((state & LayerUploaded) != 0 ? reup : first_up);
					state             = LayerUploaded;
					owner.hash[layer] = now;
					return false;
				}
				const uint64_t begin = base + slice * layer;
				const uint64_t end   = begin + slice;
				m_buffer_cache.DrainGuestReadback(begin, slice);
				const bool same  = HashGuestRange(begin, slice) == owner.hash[layer];
				uint64_t   bytes = 0;
				for (const auto& [from, to]: dirty) {
					if (to > begin && from < end) bytes += std::min(to, end) - std::max(from, begin);
				}
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
					std::printf("Shadow refresh: depth array 0x%llx (%u layers %ux%u) layer %u, rendered by the GPU, dirty "
					            "0x%llx of 0x%llx bytes (by%s%s%s%s): guest bytes %s (KYTY_SHADOW_REFRESH_FIX=0: "
					            "uploaded)\n",
					            static_cast<unsigned long long>(base), layers, info.extent.width, info.extent.height,
					            layer, static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(slice),
					            (state & LayerDirtyByWrite) != 0 ? " a CPU write" : "",
					            (state & LayerDirtyByMap) != 0 ? " a map" : "",
					            (state & LayerDirtyByUnmap) != 0 ? " an unmap" : "",
					            (state & (LayerDirtyByWrite | LayerDirtyByMap | LayerDirtyByUnmap)) == 0 ? " nothing seen" : "",
					            same ? "unchanged since: kept the GPU's" : "changed: uploaded");
					std::fflush(stdout);
				}
				if (same) {
					state = LayerRendered;
					++kept;
					return true;
				}
				state             = LayerUploaded; // (the guest's until the GPU renders it again)
				owner.hash[layer] = HashGuestRange(begin, slice);
				++changed;
				return false;
			});
			auto* pool = ShadowPoolTrace::Find(base, layers, info.extent.width, info.extent.height);
			ShadowPoolTrace::Count(pool, &ShadowPoolTrace::Pool::refresh_kept, kept);
			ShadowPoolTrace::Count(pool, &ShadowPoolTrace::Pool::refresh_changed, changed);
			ShadowPoolTrace::Count(pool, &ShadowPoolTrace::Pool::refresh_first, first_up);
			ShadowPoolTrace::Count(pool, &ShadowPoolTrace::Pool::refresh_reup, reup);
		}
	}
	if (picked.empty()) {
		LiveCounters::Add(LiveCounters::PartialUploads);
		return true;
	}
	if (image.IsGpuModified()) {
		ShadowPoolTrace::Count(ShadowPoolTrace::Find(info.data.address, layers, info.extent.width, info.extent.height),
		                       &ShadowPoolTrace::Pool::refreshed_layers, static_cast<uint32_t>(picked.size()));
	}
	// (The share is a cost limit: past it a whole upload is cheaper. Not for an array the GPU rendered into: a whole
	// upload replaces its rendered layers with guest memory, which never held them - the shadow-map pool's spot shadows,
	// kept from frame to frame, came back as stale depth: black shapes on lit floors, DepthArrayGuard.)
	if (slice * picked.size() > info.data.size / PartialUploadMaxShare && !DepthArrayGuarded(image)) {
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
	const char* refused = nullptr;
	const auto [source, source_offset] = m_buffer_cache.StageImagePieces(pieces, total, &refused);
	if (source == nullptr) {
		return PartialFail(refused == nullptr        ? "depth_stage"
		                   : refused[0] == 'g' ? "depth_stage:gpu"
		                                       : "depth_stage:map");
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
		linear = m_tiler.Detile(source->Handle(), source_offset, total, slice * picked.size(), tiles,
		                        source->Usage() == MemoryUsage::Upload);
	}
	for (auto& copy: copies) {
		copy.bufferOffset += linear.offset;
	}
	image.Upload(copies, linear.buffer, linear.offset, linear.size);
	LiveCounters::Add(LiveCounters::PartialUploads);
	LiveCounters::Add(LiveCounters::PartialUploadBytes, total);
	LogPartialUpload(image, pieces.size(), total);
	return true;
}

// The subresources over `ranges` alone, from the source a whole upload of the image takes (ObtainBufferForImage over all
// of it: the buffer cache sees the same request), detiled only where the ranges are.
bool TextureCache::UploadImageSubresources(Image& image, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	RestoreBeforeUpload(image.info.data.address, image.info.data.size); // (ram17)
	const auto& info    = image.info;
	const auto  binding = UploadBinding(image);
	if (binding == BindingType::DepthTarget || info.IsVolume()) {
		return false;
	}
	auto& cached = m_partial_plans[image.serial];
	if (cached.plan == nullptr || cached.binding != binding) {
		cached.plan    = std::make_shared<TextureTransferPlan>(BuildTextureTransfer(image, binding, TransferDirection::Upload));
		cached.binding = binding;
	}
	const auto& plan = *cached.plan;
	if (!plan.valid || plan.swap_bgra16 || plan.tiles.empty() || plan.tiles.size() != plan.regions.size()) {
		return false;
	}
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         linear_total = 0;
	for (size_t index = 0; index < plan.tiles.size(); index++) {
		auto       tile   = plan.tiles[index];
		auto       region = plan.regions[index];
		const auto begin  = info.data.address + tile.tiled_offset;
		if (!IntersectsRanges(ranges, begin, begin + tile.tiled_size)) {
			continue;
		}
		if (region.bufferOffset < tile.linear_offset) {
			return false;
		}
		const auto linear_offset = (linear_total + 255) & ~uint64_t {255};
		region.bufferOffset      = linear_offset + (region.bufferOffset - tile.linear_offset);
		tile.linear_offset       = linear_offset;
		linear_total             = linear_offset + tile.linear_size;
		tiles.push_back(tile);
		regions.push_back(region);
	}
	LiveCounters::Add(LiveCounters::PartialUploads);
	if (tiles.empty()) {
		return true; // (only padding between subresources)
	}
	const auto [source, source_offset] = m_buffer_cache.ObtainBufferForImage(info.data.address, info.data.size);
	if (source == nullptr) {
		EXIT("TextureCache: failed to obtain image upload source\n");
	}
	const auto linear = m_tiler.Detile(source->Handle(), source_offset, info.data.size, linear_total, tiles,
	                                   source->Usage() == MemoryUsage::Upload);
	for (auto& region: regions) {
		region.bufferOffset += linear.offset;
	}
	image.Upload(regions, linear.buffer, linear.offset, linear.size);
	LiveCounters::Add(LiveCounters::PartialUploadBytes, linear_total);
	return true;
}

// A whole upload of a large image, a piece of guest memory at a time, each piece from the source ObtainBufferForImage
// picks for it: staged guest bytes, the buffer holding it, or (a subresource with GPU-written bytes, a piece of its
// own) a buffer over that subresource. The whole range at once took a buffer over all of it: a texture-pool view at
// Shrine of Storms 4-2 (1568 1 MiB layers) joined the 103 small buffers inside it into one 1.57 GB buffer (3.2 s), and
// with the image and its detiling scratch (2 GiB) an 8 GB laptop paged for seconds; its shadow-map pool, inside that
// buffer, then took whole 320 MiB uploads at every layer the game streamed. The pieces detile into one reused scratch
// buffer: one each lived until the submission completed, as much again. (An image a cached buffer holds whole too: its
// pieces copy from that buffer, where its whole detile asked for a scratch buffer of the image's size - a user's 8 GB
// GPU had neither the video nor the system memory for one of 2 GiB, and the emulator exited, 10-13.)
bool TextureCache::UploadImageChunked(Image& image, const std::vector<std::pair<uint64_t, uint64_t>>* only) {
	RestoreBeforeUpload(image.info.data.address, image.info.data.size); // (ram17)
	const auto& info    = image.info;
	const auto  binding = UploadBinding(image);
	const auto  base    = info.data.address;
	if (binding == BindingType::DepthTarget || (only == nullptr && info.data.size < ChunkedUploadMin) ||
	    info.IsVolume() || base % 256 != 0) {
		return false;
	}
	// (`only`: of an image of any size, the subresources over those ranges alone.)
	const auto selected = [&](uint64_t begin, uint64_t end) {
		return only == nullptr || IntersectsRanges(*only, begin, end);
	};
	auto& cached = m_partial_plans[image.serial];
	if (cached.plan == nullptr || cached.binding != binding) {
		cached.plan    = std::make_shared<TextureTransferPlan>(
            BuildTextureTransfer(image, binding, TransferDirection::Upload));
		cached.binding = binding;
	}
	const auto& plan = *cached.plan;
	if (!plan.valid || plan.swap_bgra16 || plan.tiles.empty() || plan.tiles.size() != plan.regions.size()) {
		return false;
	}
	std::vector<size_t> order(plan.tiles.size());
	for (size_t index = 0; index < order.size(); index++) {
		if (plan.regions[index].bufferOffset < plan.tiles[index].linear_offset) {
			return false;
		}
		order[index] = index;
	}
	std::sort(order.begin(), order.end(),
	          [&](size_t left, size_t right) { return plan.tiles[left].tiled_offset < plan.tiles[right].tiled_offset; });
	const auto                       start = std::chrono::steady_clock::now();
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         pieces     = 0;
	uint64_t                         gpu_pieces = 0;
	uint64_t                         uploaded   = 0;
	for (size_t at = 0; at < order.size();) {
		const auto& first   = plan.tiles[order[at]];
		uint64_t    begin   = base + first.tiled_offset;
		uint64_t    end     = begin + first.tiled_size;
		if (!selected(begin, end)) {
			at++;
			continue;
		}
		size_t      next    = at + 1;
		const bool  written = m_buffer_cache.IsRegionGpuModified(begin, end - begin);
		while (!written && next < order.size()) {
			const auto&    tile       = plan.tiles[order[next]];
			const uint64_t tile_begin = base + tile.tiled_offset;
			const uint64_t tile_end   = tile_begin + tile.tiled_size;
			if (std::max(end, tile_end) - begin > ChunkedUploadPiece || !selected(tile_begin, tile_end) ||
			    m_buffer_cache.IsRegionGpuModified(tile_begin, tile.tiled_size)) {
				break;
			}
			end = std::max(end, tile_end);
			next++;
		}
		begin                              = begin & ~uint64_t {255};
		end                                = std::min(info.data.End(), (end + 255) & ~uint64_t {255});
		const auto [source, source_offset] = m_buffer_cache.ObtainBufferForImage(begin, end - begin);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		tiles.clear();
		regions.clear();
		uint64_t linear_total = 0;
		for (size_t k = at; k < next; k++) {
			auto       tile          = plan.tiles[order[k]];
			auto       region        = plan.regions[order[k]];
			const auto linear_offset = (linear_total + 255) & ~uint64_t {255};
			region.bufferOffset      = linear_offset + (region.bufferOffset - tile.linear_offset);
			tile.linear_offset       = linear_offset;
			tile.tiled_offset        = base + tile.tiled_offset - begin;
			linear_total             = linear_offset + tile.linear_size;
			tiles.push_back(tile);
			regions.push_back(region);
		}
		const auto linear = m_tiler.Detile(source->Handle(), source_offset, end - begin, linear_total, tiles,
		                                   source->Usage() == MemoryUsage::Upload, true);
		for (auto& region: regions) {
			region.bufferOffset += linear.offset;
		}
		image.Upload(regions, linear.buffer, linear.offset, linear.size);
		pieces++;
		gpu_pieces += written ? 1 : 0;
		uploaded += end - begin;
		at = next;
	}
	if (only != nullptr) {
		LiveCounters::Add(LiveCounters::PartialUploads);
		LiveCounters::Add(LiveCounters::PartialUploadBytes, uploaded);
		static const bool log_uploads = std::getenv("KYTY_UPLOAD_LOG") != nullptr;
		if (log_uploads) {
			std::printf("[tsc %llu] UPLOAD-RANGES addr=0x%llx size=0x%llx fmt=%u %ux%u layers=%u levels=%u tile=%u ranges=%zu "
			            "pieces=%llu gpu=%llu bytes=0x%llx serial=%llu\n",
			            static_cast<unsigned long long>(__rdtsc()), static_cast<unsigned long long>(base),
			            static_cast<unsigned long long>(info.data.size), static_cast<uint32_t>(info.guest_format),
			            info.extent.width, info.extent.height, info.resources.layers, info.resources.levels,
			            static_cast<uint32_t>(info.tile_mode), only->size(), static_cast<unsigned long long>(pieces),
			            static_cast<unsigned long long>(gpu_pieces), static_cast<unsigned long long>(uploaded),
			            static_cast<unsigned long long>(image.serial));
		}
		return true;
	}
	LiveCounters::Add(LiveCounters::FullUploads);
	LiveCounters::Add(LiveCounters::FullUploadBytes, info.data.size);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("TextureCache: chunked upload addr=0x%llx size=0x%llx layers=%u levels=%u: %llu pieces (%llu with "
		            "GPU-written bytes) in %.1f ms\n",
		            static_cast<unsigned long long>(base), static_cast<unsigned long long>(info.data.size),
		            info.resources.layers, info.resources.levels, static_cast<unsigned long long>(pieces),
		            static_cast<unsigned long long>(gpu_pieces),
		            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
	}
	return true;
}

bool TextureCache::UploadImagePartial(Image& image, uint64_t window_begin, uint64_t window_end) {
	{ // (ram17) The window inside the image (default: all of it).
		const uint64_t from = std::max(window_begin, image.info.data.address), to = std::min(window_end, image.info.data.End());
		if (to > from) RestoreBeforeUpload(from, to - from);
	}
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
		if (const auto from = std::max(begin, window_begin), to = std::min(end, window_end);
		    from >= to || !IntersectsRanges(dirty, from, to)) {
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
	// (A window's refresh leaves the rest dirty: no whole upload instead.)
	if (window_begin == 0 && window_end == UINT64_MAX && picked > info.data.size / PartialUploadMaxShare &&
	    !DepthArrayGuarded(image)) {
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
	const auto linear = m_tiler.Detile(source->Handle(), source_offset, total, linear_total, tiles,
	                                   source->Usage() == MemoryUsage::Upload);
	for (auto& region: regions) {
		region.bufferOffset += linear.offset;
	}
	image.Upload(regions, linear.buffer, linear.offset, linear.size);
	LiveCounters::Add(LiveCounters::PartialUploads);
	LiveCounters::Add(LiveCounters::PartialUploadBytes, total);
	LogPartialUpload(image, pieces.size(), total);
	if (kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 2) {
		UpdatePartialHashes(image, false);
	}
	return true;
}

// KYTY_SAMPLE_PENDING_HTILE_CLEAR (upstream 5a65bfb, PR #373; default 1; 0 off, read once): a one-layer, one-level
// depth surface whose HTile says it is fast cleared gets that clear (the fill's own value) before it is sampled. The
// clear was applied only where the image is bound as a depth target; a surface the game cleared and then only samples
// kept older contents (Demon's Souls: the 1024x512 occlusion depth, HTile-cleared by a compute fill every frame and
// sampled for its min/max pyramid: after quitting to the title and Continue, dark hard-edged slabs in the far field).
// The native and table paths leave a draw or dispatch sampling such a view to the normal path (SampleHtileClearPending).
// One line at the first clear of a surface and at every 256th.
bool TextureCache::SampleHtileClearOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_SAMPLE_PENDING_HTILE_CLEAR");
		const bool  set   = value != nullptr && value[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("Depth: a sampled depth surface's pending HTile clear applied first: %s (KYTY_SAMPLE_PENDING_HTILE_CLEAR=%s)\n",
		            set ? "on" : "off", set ? "0 turns it off" : "0");
		std::fflush(stdout);
		return set;
	}();
	return on;
}

// The HTile address of a sampled view the clear may apply to (0: none), and whether its stencil is compressed.
uint64_t TextureCache::HtileSampleMeta(const Image& image, const ImageDesc& desc, bool* stencil_compressed) {
	if (desc.type != BindingType::Texture) return 0;
	const auto& meta = desc.info.metadata.kind == ImageMetadataKind::Htile ? desc.info.metadata : image.info.metadata;
	if (meta.kind != ImageMetadataKind::Htile || !image.info.IsDepth() || image.depth_id || image.info.resources.levels != 1 ||
	    image.info.resources.layers != 1 || desc.view_info.base_layer != 0)
		return 0;
	*stencil_compressed = meta.stencil_compressed;
	return meta.range.address;
}

// (m_lock held) The clear a fill record gives a sampled depth: whether there is one, its value, whether its bit stays.
// (upstream's htile_load_decision(true, fill, true, stencil_compressed), written out with this tree's helpers.)
TextureCache::HtileSampleClear TextureCache::HtileFillClearLocked(const MetaDataInfo& record, uint64_t meta,
                                                                  bool stencil_compressed) {
	if (record.type != MetaDataInfo::Type::HTile || !record.fill_known) return {};
	const uint32_t fill          = record.fill_value;
	const bool     clears_depth  = htile_fill_clears_depth(fill);
	const bool     depth_uniform = !clears_depth && htile_fill_depth_uniform(fill, stencil_compressed);
	if (!clears_depth && !depth_uniform) return {};
	const bool stencil_meta_clear = stencil_compressed && htile_fill_clears_stencil(fill);
	return {1u, meta, htile_fill_depth_value(fill, stencil_compressed), stencil_meta_clear};
}

// (m_lock held) Whether FindTexture clears the sampled view of the HTile at `meta`. `spec` (a speculation's thread):
// as after its holes; else the record, as on the GPU thread, an answer of "not cleared" noted for the commit.
bool TextureCache::HtileWouldClearLocked(uint64_t meta, bool stencil_compressed, Spec::State* spec, HtileSampleClear* clear) {
	const auto  found  = m_surface_metas.find(meta);
	const auto* record = found != m_surface_metas.end() && found->second.type == MetaDataInfo::Type::HTile ? &found->second
	                                                                                                        : nullptr;
	*clear = record != nullptr ? HtileFillClearLocked(*record, meta, stencil_compressed) : HtileSampleClear {};
	const bool pending = record != nullptr && (record->clear_mask & 1u) != 0 && (record->sample_depth_done & 1u) == 0;
	if (spec == nullptr) return clear->slices != 0 && pending;
	if (const auto known = spec->KnownMeta(meta, 0); known >= 0) return known == 1;
	if (record != nullptr && (record->clear_mask & 1u) != 0) return clear->slices != 0 && pending;
	spec->NoteMetaRead(meta, 0);
	return false;
}

TextureCache::HtileSampleClear TextureCache::HtileSampleClearLocked(const Image& image, const ImageDesc& desc) {
	bool       stencil_compressed = false;
	const auto meta               = HtileSampleMeta(image, desc, &stencil_compressed);
	if (meta == 0) return {};
	HtileSampleClear clear;
	if (CommandScheduler::ThreadRecorder() != nullptr) {
		// (A speculation's recorder thread clears nothing: where FindTexture would, the work is the GPU thread's.)
		if (HtileWouldClearLocked(meta, stencil_compressed, Spec::Current(), &clear))
			throw CommandScheduler::RecorderRefusal {"sampled depth's HTile clear"};
		return {};
	}
	return HtileWouldClearLocked(meta, stencil_compressed, nullptr, &clear) ? clear : HtileSampleClear {};
}

bool TextureCache::SampleHtileClearPending(const Image& image, const ImageDesc& desc, uint64_t* meta, bool* consume) {
	if (!SampleHtileClearOn()) return false;
	bool       stencil_compressed = false;
	const auto address            = HtileSampleMeta(image, desc, &stencil_compressed); // (before the lock)
	if (address == 0) return false;
	*meta      = address;
	auto* spec = Spec::Current();
	if (spec == nullptr && !MetaFilterMayHold(address)) return false;
	std::scoped_lock lock {m_lock};
	HtileSampleClear clear;
	const bool       would = HtileWouldClearLocked(address, stencil_compressed, spec, &clear);
	*consume               = clear.slices == 0 || !clear.keep;
	return would;
}

void TextureCache::ApplyHtileSampleClear(ImageId id, const HtileSampleClear& clear) {
	const auto found = m_surface_metas.find(clear.meta);
	if (found == m_surface_metas.end() || found->second.type != MetaDataInfo::Type::HTile) return;
	auto&          record = found->second;
	const auto&    image  = m_slot_images[id];
	vk::ClearValue value {};
	value.depthStencil = vk::ClearDepthStencilValue {clear.value, 0};
	ClearImage(m_scheduler.Current(), id, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}, value);
	if (clear.keep) record.sample_depth_done |= 1u; // (the target bind still clears the stencil)
	else record.clear_mask &= ~1u;
	if (const auto clears = ++record.sample_clears; clears == 1 || clears % 256 == 0) {
		std::printf("Depth: pending HTile clear applied before a sample at 0x%llx (%s %ux%u, value %g%s): %llu times\n",
		            static_cast<unsigned long long>(image.info.data.address), vk::to_string(image.backing.format).c_str(),
		            image.info.extent.width, image.info.extent.height, static_cast<double>(clear.value),
		            clear.keep ? ", depth only: the stencil's left to the bind" : "", static_cast<unsigned long long>(clears));
		std::fflush(stdout);
	}
}

void TextureCache::PrepareDccClear(ImageId id, const ImageDesc& desc) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc) {
		return;
	}
	auto&      image       = m_slot_images[id];
	bool       changed     = !(image.info.metadata == desc.info.metadata);
	if (changed) ++image.validity;
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
	DeferredImageProtects protects {m_buffer_cache, m_page_manager};
	protects.Track([&] { TrackImage(id); });
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
		m_buffer_cache.ApplyImageProtects(); // (the hash reads the pages: a write after it has to fault)
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

// A depth target binding writes only the layers its view covers. Of a depth array that is also dirty in other layers,
// only the dirty layers it covers are uploaded; the others stay dirty (their pages released: no write faults) until a
// binding covers them. The game's 80-layer shadow-map pool (320 MiB) renders shadows into layers 0-4 while its texture
// streamer tiles textures into free layers from the CPU: each shadow pass uploaded the layer being streamed (4 MiB per
// binding, 145 times = 600 MiB in one 101 ms frame at the Shrine).
bool TextureCache::RefreshDepthLayers(ImageId id, const ImageViewInfo& view) {
	auto&       image  = m_slot_images[id];
	const auto& info   = image.info;
	const auto  layers = info.resources.layers;
	if (!image.IsPartiallyCpuDirty() || image.IsBufferModified() || image.IsStencilModified() ||
	    image.IsMaybeCpuDirty() || image.track_addr != info.data.address || image.track_addr_end != info.data.End() ||
	    layers < 2 || view.layer_count == 0 ||
	    view.base_layer >= layers || view.layer_count > layers - view.base_layer ||
	    view.layer_count == layers || info.data.size % layers != 0 ||
	    (info.data.size / layers) % TRACKER_PAGE_SIZE != 0 || info.data.address % TRACKER_PAGE_SIZE != 0 ||
	    info.metadata.compression != VideoOutCompression::Uncompressed ||
	    UploadBinding(image) != BindingType::DepthTarget) {
		return false;
	}
	const uint32_t first = view.base_layer;
	const uint32_t last  = view.base_layer + view.layer_count;
	const uint64_t slice = info.data.size / layers;
	const uint64_t begin = info.data.address + slice * first;
	const uint64_t end   = info.data.address + slice * last;
	if (IntersectsRanges(image.CpuDirtyRanges(), begin, end)) {
		g_partial_fail = "";
		if (!UploadDepthPartial(image, first, last)) {
			return false;
		}
		// As FinishRefresh, for these layers: watched again before they count as clean.
		RetrackHoles(image, begin, end);
		image.RefreshRangeComplete(begin, end);
	}
	return true;
}

bool TextureCache::LevelRefreshCandidate(const Image& image) {
	// Textures the CPU writes and the GPU only samples (one written as a target or storage keeps whole refreshes),
	// without layers (a level is one range of bytes), on whole pages.
	const auto& info = image.info;
	return kyty_local_texture_levels_mode.load(std::memory_order_relaxed) != 0 &&
	       kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 1 && PartialDirtyCandidate(image) &&
	       image.registered && !image.IsGpuModified() && !image.IsBufferModified() && !image.IsStencilModified() &&
	       !image.usage.storage && !image.usage.render_target && !image.usage.depth_target && !image.usage.video_out &&
	       !image.binding.is_target && !info.IsDepth() && !info.HasMetadata() && info.resources.layers == 1 &&
	       info.resources.levels > 1 && info.resources.levels <= info.mip_layout.size() &&
	       info.data.address % TRACKER_PAGE_SIZE == 0 && info.data.End() % TRACKER_PAGE_SIZE == 0;
}

bool TextureCache::ViewLevelBytes(const Image& image, const ImageViewInfo& view, uint64_t& begin, uint64_t& end) {
	const auto key = view.base_level | view.level_count << 8u;
	if (image.level_view != key) {
		image.level_view  = key;
		image.level_begin = 0;
		image.level_end   = 0;
		const auto& info   = image.info;
		const auto  levels = info.resources.levels;
		if (view.base_level != 0 && levels <= info.mip_layout.size() && view.base_level < levels &&
		    view.level_count != 0 && view.level_count <= levels - view.base_level) {
			uint64_t first = UINT64_MAX, last = 0;
			for (uint32_t level = view.base_level; level < view.base_level + view.level_count; ++level) {
				first = std::min(first, info.mip_layout[level].offset);
				last  = std::max(last, info.mip_layout[level].offset + info.mip_layout[level].size);
			}
			// Every level whose bytes the range touches is in it whole (the levels of the mip tail share one block).
			for (bool grew = true; grew;) {
				grew = false;
				for (uint32_t level = 0; level < levels; ++level) {
					const auto& mip = info.mip_layout[level];
					if (mip.size == 0 || mip.offset >= last || mip.offset + mip.size <= first) continue;
					if (mip.offset < first || mip.offset + mip.size > last) {
						first = std::min(first, mip.offset);
						last  = std::max(last, mip.offset + mip.size);
						grew  = true;
					}
				}
			}
			// Levels left out (the range is not the whole image), on pages of their own: the range ends on page
			// boundaries.
			const auto from = info.data.address + first;
			const auto to   = info.data.address + std::min(last, info.data.size);
			if (first < last && from < to && (first != 0 || last < info.data.size) && from % TRACKER_PAGE_SIZE == 0 &&
			    to % TRACKER_PAGE_SIZE == 0) {
				image.level_begin = from;
				image.level_end   = to;
			}
		}
	}
	begin = image.level_begin;
	end   = image.level_end;
	return end != 0;
}

bool TextureCache::DirtyForView(const Image& image, const ImageDesc& desc) {
	uint64_t begin = 0, end = 0;
	return !image.IsPartiallyCpuDirty() || desc.type != BindingType::Texture || !LevelRefreshCandidate(image) ||
	       !ViewLevelBytes(image, desc.view_info, begin, end) || IntersectsRanges(image.CpuDirtyRanges(), begin, end);
}

// A sampled view reads its levels only. The game streams a texture in from its smallest levels and its T# starts at the
// first level loaded (2-2 walk: 185 first uploads of textures sampled from level 3, which of a 4096x4096 BC texture is
// 0.4 of its 21 MiB; the larger levels are bytes the game has not written yet). A whole refresh uploaded all of it, at
// first use and again after each level that came in: turning into a new view, ~1 GB in three frames (50-68 ms, the GPU
// behind on the PCIe transfers). Of a texture dirty below the view's levels, only the dirty bytes of its levels (and of
// the levels sharing their pages) are uploaded; the others stay dirty, their pages released (no write faults while the
// game streams them in), until a view covers them (the game lowers the T#'s base level once a level is loaded).
bool TextureCache::RefreshTextureLevels(ImageId id, const ImageViewInfo& view) {
	auto&       image = m_slot_images[id];
	NativeGuardDirty(image); // (N2+N3)
	const auto& info  = image.info;
	uint64_t    begin = 0, end = 0;
	if (!image.IsDefinitelyCpuDirty() || image.IsMaybeCpuDirty() || !LevelRefreshCandidate(image) ||
	    !ViewLevelBytes(image, view, begin, end)) {
		return false;
	}
	const bool whole = !image.IsPartiallyCpuDirty();
	if (whole ? image.IsTracked()
	          : image.track_addr != info.data.address || image.track_addr_end != info.data.End()) {
		return false;
	}
	if (!whole && !IntersectsRanges(image.CpuDirtyRanges(), begin, end)) {
		return true; // its levels are current
	}
	// Watched before the guest bytes are read (as RefreshImage's TrackImage): a write after the read faults.
	if (whole) {
		image.track_addr     = info.data.address;
		image.track_addr_end = info.data.End();
		WatchImagePages<true>(m_page_manager, image, begin, end - begin);
		if (info.data.address < begin) Image::AddRange(image.untracked_holes, info.data.address, begin);
		if (end < info.data.End()) Image::AddRange(image.untracked_holes, end, info.data.End());
		image.NarrowCpuDirty();
		LiveCounters::Add(LiveCounters::LevelSkippedBytes, info.data.size - (end - begin));
	} else {
		RetrackHoles(image, begin, end);
	}
	g_partial_fail = "";
	if (!UploadImagePartial(image, begin, end)) {
		return false; // (RefreshImage watches the rest again and uploads what is dirty)
	}
	image.staged_ring = nullptr;
	image.RefreshRangeComplete(begin, end);
	LiveCounters::Add(LiveCounters::LevelRefreshes);
	return true;
}

// (SUBIDA-REP3) local/subida-rep.h. Tras una subida ENTERA por el camino normal (la imagen estaba sucia y RefreshImage
// la dejo limpia): hash por capa de sus bytes del juego = lo que tiene la GPU. Solo texturas grandes de 1 nivel por capas.
void TextureCache::MarcarContenido(const Image& image) {
	const auto& info = image.info;
	if (info.data.size < SubidaRep::MinBytes() || !SubidaRep::On() || image.backing.image == nullptr ||
	    image.IsCpuDirty() || image.IsBufferModified() || image.IsGpuModified() || info.resources.levels != 1 ||
	    info.resources.layers < 2 || info.data.size % info.resources.layers != 0 ||
	    (info.data.size / info.resources.layers) % TRACKER_PAGE_SIZE != 0 || info.data.address % TRACKER_PAGE_SIZE != 0 ||
	    !image.usage.texture || image.usage.storage || image.usage.render_target || image.usage.depth_target ||
	    image.usage.video_out || info.IsDepth() || info.IsVolume()) {
		return;
	}
	auto&          s      = SubidaRep::S();
	const int64_t  t0     = SubidaRep::NowUs();
	const uint32_t layers = info.resources.layers;
	const uint64_t slice  = info.data.size / layers;
	auto&          c      = s.capas[image.serial];
	c.key                 = SubidaRep::Key(image.backing.image);
	c.hash.resize(layers);
	for (uint32_t l = 0; l < layers; ++l) c.hash[l] = HashGuestRange(info.data.address + slice * l, slice);
	// Una escritura del juego durante el hash ensucia la imagen: esa base ya no es fiable.
	if (image.IsCpuDirty()) s.capas.erase(image.serial);
	s.bases++;
	s.base_us += static_cast<uint64_t>(SubidaRep::NowUs() - t0);
}

// (SUBIDA-REP3) Refresco de una textura grande por capas (1 nivel), solo muestreada y sucia solo por la CPU, con hashes
// por capa de su ultima subida: se vigilan sus paginas, se hashean las capas sucias y se suben solo las que cambiaron.
// Nunca aplaza: true = imagen limpia (como RefreshImage); false = RefreshImage (camino de siempre). Sin budget, sin
// espera y sin MoveImageStateEpoch (la version que aplazaba dejaba la imagen sucia y costaba 60 ms por fotograma en los
// draws por tabla, log 074406). Log 054756: arrays de 360 capas (135 y 180 MB) subidos enteros, 97 + 65 ms.
bool TextureCache::SubidaRepartida(ImageId id) {
	auto&       image = m_slot_images[id];
	const auto& info  = image.info;
	if (!image.IsDefinitelyCpuDirty() || image.IsMaybeCpuDirty() || info.data.size < SubidaRep::MinBytes() ||
	    !SubidaRep::On()) {
		return false; // (camino normal de los draws: termina aqui)
	}
	auto&      s     = SubidaRep::S();
	const auto known = s.capas.find(image.serial);
	if (known == s.capas.end()) return false;
	const uint32_t layers = info.resources.layers;
	if (known->second.key != SubidaRep::Key(image.backing.image) || known->second.hash.size() != layers) {
		s.capas.erase(known);
		return false;
	}
	if (Spec::Current() != nullptr || CommandScheduler::ThreadRecorder() != nullptr || AsyncUpload::t_worker ||
	    CommandScheduler::InDeferredOperation()) {
		return false;
	}
	if (info.resources.levels != 1 || layers < 2 || info.data.size % layers != 0 ||
	    (info.data.size / layers) % TRACKER_PAGE_SIZE != 0 || info.data.address % TRACKER_PAGE_SIZE != 0 ||
	    !image.registered || !image.usage.texture || image.usage.storage || image.usage.render_target ||
	    image.usage.depth_target || image.usage.video_out || image.binding.is_target || image.depth_id ||
	    info.IsDepth() || info.HasStencil() || info.HasMetadata() || info.IsVolume() ||
	    info.metadata.compression != VideoOutCompression::Uncompressed || image.IsGpuModified() ||
	    image.IsBufferModified() || image.IsStencilModified() || image.IsSparse() || RtNativo::Es(image) ||
	    kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) != 1 || !PartialDirtyCandidate(image) ||
	    TextureOffload::Overlaps(info.data.address, info.data.size) ||
	    TextureOffload::IsRestoring(info.data.address, info.data.size)) {
		s.capas.erase(known);
		return false;
	}
	const bool whole = !image.IsPartiallyCpuDirty();
	if (image.IsTracked() && (image.track_addr != info.data.address || image.track_addr_end != info.data.End())) {
		return false;
	}
	if (!whole && !image.IsTracked()) return false;
	const int64_t  t0    = SubidaRep::NowUs();
	const uint64_t slice = info.data.size / layers;
	if (whole) {
		// Sus paginas, vigiladas antes de leerlas (RetrackHoles); lo no vigilado, como hueco.
		if (!image.IsTracked()) {
			image.track_addr     = info.data.address;
			image.track_addr_end = info.data.End();
			Image::AddRange(image.untracked_holes, info.data.address, info.data.End());
		}
		image.NarrowCpuDirty();
	}
	// Capas sucias (en capas enteras), con sus paginas vigiladas ANTES de leer: una escritura despues ensucia de nuevo.
	std::vector<std::pair<uint64_t, uint64_t>> sucias;
	for (const auto& [rb, re]: image.CpuDirtyRanges()) {
		const uint64_t b = info.data.address + (std::max(rb, info.data.address) - info.data.address) / slice * slice;
		const uint64_t e = std::min(info.data.End(),
		                            info.data.address + (std::min(re, info.data.End()) - info.data.address + slice - 1) /
		                                                    slice * slice);
		if (b < e) Image::AddRange(sucias, b, e);
	}
	for (const auto& [b, e]: sucias) RetrackHoles(image, b, e);
	auto&    hashes   = known->second.hash;
	uint64_t revisadas = 0, subidas = 0;
	bool     failed    = false;
	for (const auto& [b, e]: sucias) {
		uint64_t run_begin = 0, run_end = 0;
		std::vector<std::pair<uint32_t, uint64_t>> nuevas;
		const auto flush = [&]() {
			if (run_begin == run_end) return true;
			g_partial_fail = "";
			if (!UploadImagePartial(image, run_begin, run_end)) return false;
			for (const auto& [l, h]: nuevas) hashes[l] = h;
			nuevas.clear();
			run_begin = run_end = 0;
			return true;
		};
		for (uint64_t at = b; at < e; at += slice) {
			const auto     l = static_cast<uint32_t>((at - info.data.address) / slice);
			const uint64_t h = HashGuestRange(at, slice);
			revisadas++;
			if (h == hashes[l]) {
				if (!flush()) { failed = true; break; }
				continue;
			}
			subidas++;
			nuevas.emplace_back(l, h);
			if (run_begin == run_end) run_begin = at;
			run_end = at + slice;
		}
		if (failed || !flush()) { failed = true; break; }
	}
	const uint64_t us = static_cast<uint64_t>(SubidaRep::NowUs() - t0);
	s.us += us;
	if (us > s.max_us) s.max_us = us;
	if (failed) {
		s.capas.erase(image.serial); // RefreshImage sube lo que siga sucio y MarcarContenido rehace la base
		return false;
	}
	image.staged_ring = nullptr;
	for (const auto& [b, e]: sucias) {
		if (image.IsPartiallyCpuDirty()) image.RefreshRangeComplete(b, e);
	}
	if (image.IsCpuDirty()) { // (no deberia: rangos fuera de las capas) el resto por el camino de siempre
		s.capas.erase(image.serial);
		return false;
	}
	LiveCounters::Add(LiveCounters::LevelRefreshes);
	s.refrescos++;
	s.revisadas += revisadas;
	s.subidas += subidas;
	s.bytes_evitados += (revisadas - subidas) * slice;
	SubidaRep::Resumen();
	return true;
}

void TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	// (Not an image the GPU draws or writes to at that address: memory it shares with the stencil plane (a pool of
	// targets reused while an area loads) made it the depth image's from then on, and the next draw to it ended the
	// emulator, "color target requires rediscovery before final acquisition" (issue #94, fast travel to 1-1). A record of
	// its own beside it.)
	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != stencil.address) continue;
		if (!owner->usage.render_target && !owner->usage.depth_target && !owner->usage.storage) {
			association = id;
		} else if (static std::atomic<uint32_t> reports {0}; reports.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::printf("TextureCache: a %s target at 0x%llx shares its memory with a depth image's stencil: kept apart\n",
			            owner->usage.depth_target ? "depth" : "color", static_cast<unsigned long long>(stencil.address));
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
	if (record.depth_id != depth_id) ++record.validity;
	record.depth_id = depth_id;
	if (std::find(m_stencil_associations.begin(), m_stencil_associations.end(), association) ==
	    m_stencil_associations.end()) {
		m_stencil_associations.push_back(association);
	}
}

bool TextureCache::HoldsImage(const ImageInfo& info) {
	if (info.data.Empty()) return false;
	std::scoped_lock          lock {m_lock};
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(info.data.address, info.data.size, pages)) return false;
	const auto* owners = m_image_page_table.Find(pages.first);
	if (owners == nullptr) return false;
	bool held = false;
	owners->ForEach([&](ImageId id) {
		const auto* image = m_slot_images.try_get(id);
		held              = held || (image != nullptr && image->registered && SameBacking(image->info, info, false));
	});
	return held;
}

// descriptors.cpp HoldLayerRange: `info` holds the layers of a one-level array a sampled view reads, slices one after
// another. An image of the same shape whose layers lie on the same grid of slices holds the same texture in the same
// bytes, so it can serve the view: one holding all these layers (`info` becomes its own, `first` the view's first layer
// in it), or else the ones over part of them grow into one image of all their layers, made once: their layers copied on
// the GPU, the others uploaded, and they go. The game's texture pools are sampled through views from a few first layers
// to a last one that grows as more streams in: each became an image of its own (0.8-3.6 GB a piece on a 16 GB RX 9070,
// with the whole pool's beside them), each larger last layer a new image uploaded whole beside the old one (3.2 s
// frames), until the GPU's memory was full and the emulator exited (issue #49). Without such images `info` stays.
void TextureCache::HoldLayers(ImageInfo& info, uint32_t& first) {
	first             = 0;
	const auto layers = info.resources.layers;
	if (info.data.Empty() || layers == 0 || info.resources.levels != 1 || info.data.size % layers != 0 ||
	    info.mip_layout[0].offset != 0 || info.mip_layout[0].size != info.data.size) {
		return;
	}
	const uint64_t slice   = info.data.size / layers;
	const auto     on_grid = [&](const ImageInfo& held) {
		return held.resources.levels == 1 && held.resources.layers != 0 &&
		       held.data.size == slice * held.resources.layers && held.samples == info.samples && held.type == info.type &&
		       held.pixel_format == info.pixel_format && held.guest_format == info.guest_format &&
		       held.extent == info.extent && held.pitch == info.pitch && held.bytes_per_block == info.bytes_per_block &&
		       held.tile_mode == info.tile_mode && held.bgra16 == info.bgra16 && !held.HasMetadata() &&
		       !held.HasStencil() && held.mip_layout[0].offset == 0 && held.mip_layout[0].size == held.data.size &&
		       held.mip_layout[0].pitch == info.mip_layout[0].pitch &&
		       held.mip_layout[0].height == info.mip_layout[0].height &&
		       (std::max(held.data.address, info.data.address) - std::min(held.data.address, info.data.address)) % slice ==
		           0;
	};
	const auto           view_address = info.data.address;
	ImageInfo            grown        = info;
	std::vector<ImageId> parts;
	{
		std::scoped_lock lock {m_lock};
		// (One holding them all is over their first page: the lookup of every draw that binds them looks there alone.)
		ImageId                   holder {};
		ImagePageTable::PageRange pages {};
		if (!ImagePageTable::TryGetPageRange(info.data.address, info.data.size, pages)) return;
		if (const auto* owners = m_image_page_table.Find(pages.first)) {
			owners->ForEach([&](ImageId id) {
				const auto* image = m_slot_images.try_get(id);
				if (image == nullptr || !image->registered || !on_grid(image->info) ||
				    image->info.data.address > info.data.address || image->info.data.End() < info.data.End()) {
					return;
				}
				if (!holder || image->info.data.size < m_slot_images[holder].info.data.size) holder = id;
			});
		}
		if (holder) {
			info  = m_slot_images[holder].info;
			first = static_cast<uint32_t>((view_address - info.data.address) / slice);
			return;
		}
		uint64_t begin = info.data.address;
		uint64_t end   = info.data.End();
		for (const auto id: FindImagesInRegion(info.data.address, info.data.size, false)) {
			const auto* image = m_slot_images.try_get(id);
			if (image == nullptr || !image->registered || !on_grid(image->info) || !GuestTexture(*image)) continue;
			parts.push_back(id);
			begin = std::min(begin, image->info.data.address);
			end   = std::max(end, image->info.data.End());
		}
		if (parts.empty()) return;
		grown.data               = {begin, end - begin};
		grown.resources.layers   = static_cast<uint32_t>((end - begin) / slice);
		grown.mip_layout[0].size = grown.data.size;
		if ((end - begin) / slice > m_graphics.GetPhysicalDeviceProperties().limits.maxImageArrayLayers ||
		    RecentlyFailed(grown)) {
			return;
		}
	}
	// (Guest memory current before its bytes are uploaded, as FindImage does.)
	m_buffer_cache.DrainGuestReadback(grown.data.address, grown.data.size);
	std::scoped_lock lock {m_lock};
	const auto       id = InsertImage(grown, true);
	if (!id) return;
	if (const auto* part = m_slot_images.try_get(parts.front())) m_slot_images[id].usage = part->usage;
	DeferredImageProtects protects {m_buffer_cache, m_page_manager};
	protects.Track([&] { TrackImage(id); });
	std::vector<std::pair<uint64_t, uint64_t>> uploads {{grown.data.address, grown.data.End()}};
	for (const auto part: parts) {
		const auto* image = m_slot_images.try_get(part);
		if (image == nullptr || !image->registered) continue;
		CopyImageMip(id, part, 0, static_cast<uint32_t>((image->info.data.address - grown.data.address) / slice));
		// (One whose pages a write may have touched since gives its layers from guest memory instead.)
		if (const auto& copied = m_slot_images[part]; !copied.IsCpuDirty()) {
			Image::SubtractRange(uploads, copied.info.data.address, copied.info.data.End());
		}
	}
	// The layers no image held: uploaded alone. (Otherwise it stays dirty as made, and its first use uploads all of it.)
	if (auto& image = m_slot_images[id]; uploads.empty() || UploadImageChunked(image, &uploads)) {
		image.staged_ring = nullptr;
		if (image.IsCpuDirty()) FinishRefresh(image);
	}
	for (const auto part: parts) {
		auto* image = m_slot_images.try_get(part);
		if (image == nullptr || !image->registered) continue;
		if (image->binding.is_bound || image->binding.is_target) image->binding.needs_rebind = true;
		FreeImage(part, "layer-range");
	}
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("TextureCache: layers 0x%llx+0x%llx (%u) held with %zu image(s) beside them as one of %u layers "
		            "(0x%llx)\n",
		            static_cast<unsigned long long>(view_address), static_cast<unsigned long long>(info.data.size), layers,
		            parts.size(), grown.resources.layers, static_cast<unsigned long long>(grown.data.address));
		std::fflush(stdout);
	}
	info  = grown;
	first = static_cast<uint32_t>((view_address - grown.data.address) / slice);
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format, bool may_fail) {
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
	if (desc.info.data.size) TakeImageCpuWrites(m_page_manager, *this, desc.info.data.address, desc.info.data.size);
	if (desc.info.stencil.size) TakeImageCpuWrites(m_page_manager, *this, desc.info.stencil.address, desc.info.stencil.size);
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
		m_alias_insert = false;
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
		// (W14, KYTY_VISTA_FORMATO, por defecto encendido; =0 apagado) Log 20261125-120123 [vram-desg]: 12 imagenes
		// duplicadas en la misma direccion (45 MiB): una textura que pide su formato exacto (conversion en el shader) no
		// tomaba la imagen que ya habia con otro formato de la misma clase de bits y se hacia otra copia en VRAM (subida
		// desde la RAM del juego: con datos viejos si la GPU habia pintado la primera). Las imagenes de color se crean con
		// VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT (ImageCreateFlags): ahora es una vista con el formato pedido de la misma
		// imagen (mismos bits, una sola copia). Solo texturas muestreadas, misma forma/tiling/tamano, sin profundidad.
		bool vista_formato = false;
		if (!result && exact_format && desc.type == BindingType::Texture && VistaFormato::On()) {
			for (const auto id: candidates) {
				const auto& image = m_slot_images[id];
				if (image.info.IsDepth() || image.IsSparse() ||
				    !(image.backing.flags & vk::ImageCreateFlagBits::eMutableFormat) ||
				    !SameBacking(image.info, desc.info, false) || !image.info.resources.Contains(desc.info.resources)) {
					continue;
				}
				result        = id;
				vista_formato = true;
			}
			if (vista_formato) {
				desc.view_info.format = desc.info.pixel_format;
				VistaFormato::Cuenta(m_slot_images[result]);
			}
		}
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
			if (exact_format && !vista_formato && resolved.info.pixel_format != desc.info.pixel_format) {
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
				// (A sampled view of a guest texture in a shape it does not hold, which would not hold the texture's
				// own either: an image of its own, the other view keeps this one, GuestTexture. One holding it replaces
				// it as before: requests of the old shape find it.)
				if (resolved.info.data.address == desc.info.data.address && GuestTextureRequest(desc.type, desc.info) &&
				    GuestTexture(resolved) && !desc.info.resources.Contains(resolved.info.resources)) {
					NoteGuestAlias(desc.info, resolved);
				} else {
					FreeImage(result, "find-resources");
				}
				result = {};
			}
		}
		// (RT-ALIAS-MEM) Un RT ligado a memoria compartida que se leyo despues de que otro la pisara: se recrea con memoria
		// propia (su rango queda fuera del aliasing, Image::MemAliasBlocked).
		if (result && m_slot_images[result].MemAliasBad()) {
			FreeImage(result, "rt-alias-mem-mala");
			result = {};
		}
		// (M7, TEX-DEDUP) Copy-on-write: un miembro de una VkImage compartida que se va a escribir (CPU, bufer de la GPU,
		// GPU, RT, storage, profundidad) sale antes de usarse; abajo se crea con su propia VkImage. Los otros siguen igual.
		if (result) {
			auto& compartida = m_slot_images[result];
			if (compartida.SharedWithOthers() &&
			    (desc.type != BindingType::Texture || compartida.IsCpuDirty() || compartida.IsBufferModified() ||
			     compartida.IsGpuModified() || compartida.IsStencilModified() || !GuestTexture(compartida) ||
			     m_buffer_cache.HasGpuDirtyBytes(compartida.info.data.address, compartida.info.data.size))) {
				FreeImage(result, "tex-dedup-cow");
				Image::NoteDedupSeparated();
				result = {};
			}
		}
		if (!result) {
			// (No image: the caller binds a null one. Memory it could not have is not asked for at every draw.)
			if (may_fail && RecentlyFailed(desc.info)) return {};
			// (M7, TEX-DEDUP) Mismo contenido ya en la VRAM en otra direccion: la imagen nueva es alias de esa VkImage.
			uint64_t dedup_hash = 0;
			ImageId  dedup_src {};
			if (desc.type == BindingType::Texture && TexDedup::On() && TexDedup::Apta(desc.info) && Spec::Current() == nullptr &&
			    CommandScheduler::ThreadRecorder() == nullptr &&
			    !TextureOffload::Overlaps(desc.info.data.address, desc.info.data.size) &&
			    !m_buffer_cache.HasGpuDirtyBytes(desc.info.data.address, desc.info.data.size)) {
				dedup_hash = HashGuestRange(desc.info.data.address, desc.info.data.size);
				dedup_src  = TexDedupFuente(desc.info, dedup_hash);
			}
			if (dedup_src) Image::SetShareSource(&m_slot_images[dedup_src]);
			// (RT-ALIAS-MEM) La imagen viva mas grande de la misma arena RT que contiene todo su rango: se liga a su memoria.
			if (!dedup_src && desc.type != BindingType::Texture && desc.type != BindingType::VideoOut && RtAliasMemOn() &&
			    !desc.info.data.Empty() && Spec::Current() == nullptr && CommandScheduler::ThreadRecorder() == nullptr) {
				const uint64_t a     = desc.info.data.address;
				const uint64_t e     = desc.info.data.End();
				const int      arena = Loader::DemonsSoulsMemMap::RtArenaOf(a, e);
				if ((arena == 1 || arena == 2) && !Image::MemAliasBlocked(a, desc.info.data.size)) {
					Image*   host = nullptr;
					uint64_t best = 0;
					for (const auto id: FindImagesInRegion(a, desc.info.data.size, false)) {
						auto* c = m_slot_images.try_get(id);
						if (c == nullptr || !c->registered || !c->mem_alias_capable || c->backing.image == nullptr ||
						    c->backing.allocation == nullptr ||
						    c->IsSparse() || c->share != nullptr || c->MemAliasBad() || c->info.data.address > a ||
						    c->info.data.End() < e || c->info.data.size <= best)
							continue;
						if (Loader::DemonsSoulsMemMap::RtArenaOf(c->info.data.address, c->info.data.End()) != arena ||
						    Image::MemAliasBlocked(c->info.data.address, c->info.data.size))
							continue;
						best = c->info.data.size;
						host = c;
					}
					Image::SetMemAliasHost(host, host != nullptr ? a - host->info.data.address : 0, true, a);
				}
			}
			result = InsertImage(desc.info, may_fail);
			Image::SetShareSource(nullptr);
			Image::SetMemAliasHost(nullptr, 0, false, 0);
			if (result && dedup_hash != 0) result = TexDedupAlta(result, desc.info, dedup_hash, may_fail);
			// KYTY_VRAM_SAVER (a GPU short of video memory only): a texture the GPU could not make was a null (black)
			// one; images unused for 16 flips go (ReclaimImageMemory) and it is tried once more. (Not on a
			// speculation's recorder: it cannot wait for the GPU.)
			if (!result && may_fail && m_graphics.VramSaver() && CommandScheduler::ThreadRecorder() == nullptr &&
			    !AsyncUpload::t_worker && !CommandScheduler::InDeferredOperation()) {
				if (ReclaimImageMemory(desc.info.data.size, false) != 0) {
					std::erase_if(m_failed_images, [&](const FailedImage& failed) {
						return failed.address == desc.info.data.address;
					});
					result = InsertImage(desc.info, may_fail);
					if (result) VramSaverAdd(VramRescued, 1);
				}
			}
			if (!result) return {};
			auto& inserted       = m_slot_images[result];
			inserted.guest_alias = m_alias_insert;
			m_alias_insert       = false;
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
		}
		if (m_guest_aliases != 0 && desc.type != BindingType::Texture) {
			FreeGuestAliases(result);
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
	TakeImageCpuWrites(m_page_manager, *this, desc.info.data.address, desc.info.data.size);
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
	TakeImageCpuWrites(m_page_manager, *this, desc.info.data.address, desc.info.data.size);
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
	// (The bucket bits are written under m_lock: a clear one needs neither the lock nor the map.)
	if (!m_lock.ReadShared([&] { return MayStartAt(address); })) return false;
	std::scoped_lock lock {m_lock};
	return MayStartAt(address) && m_image_starts.contains(address);
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
	TakeImageCpuWrites(m_page_manager, *this, address, size);
	std::scoped_lock lock {m_lock};
	if (kyty_local_image_granules_mode.load(std::memory_order_relaxed) == 2 && !MayHaveImages(address, 1)) {
		return {};
	}
	// Only registered images starting at `address` qualify.
	if (!MayStartAt(address) || !m_image_starts.contains(address)) {
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
	if (image.info.resources.layers >= 6 && (image.info.IsDepth() || image.depth_array_alias)) {
		ShadowPoolTrace::Count(ShadowPoolTrace::Find(image.info.data.address, image.info.resources.layers,
		                                             image.info.extent.width, image.info.extent.height),
		                       desc.type == BindingType::Storage ? &ShadowPoolTrace::Pool::storage
		                                                         : &ShadowPoolTrace::Pool::samples);
	}
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	// (KYTY_SAMPLE_PENDING_HTILE_CLEAR: decided before the upload, a speculation's recorder refused before anything
	// changed; applied after the upload, which it overrides.)
	const auto htile = desc.type == BindingType::Texture && !image.info.data.Empty() && SampleHtileClearOn()
	                       ? HtileSampleClearLocked(image, desc)
	                       : HtileSampleClear {};
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
		GestorTex::Reclasifica(image); // (M7) escrita por la GPU: sin respaldo
	}
	if (!image.info.data.Empty()) {
		PrepareDccClear(id, desc);
		if (desc.type != BindingType::Texture ||
		    (!RefreshTextureLevels(id, desc.view_info) && !SubidaRepartida(id))) { // (SUBIDA-REP)
			const bool sucia = desc.type == BindingType::Texture && image.IsCpuDirty(); // (SUBIDA-REP3)
			RefreshImage(id);
			if (sucia) MarcarContenido(image); // (SUBIDA-REP3: base de hashes tras una subida entera)
		}
		if (htile.slices != 0) ApplyHtileSampleClear(id, htile);
	}
	// (NEGRAS-LOG c) Vista del draw que incluye un mip nativo sin memoria ligada (se lee como cero = negro).
	if (TexNativa::NegrasLog() && desc.type == BindingType::Texture && SparseIsPerLevel(image.backing)) {
		const uint32_t first = desc.view_info.base_level;
		const uint32_t last  = std::min(first + desc.view_info.level_count, image.info.resources.levels);
		for (uint32_t level = first; level < last; level++) {
			if (SparseLevelBound(image.backing, level)) continue;
			static std::unordered_set<uint64_t> seen;
			if (seen.size() < 100000 && seen.insert(image.serial << 5u | level).second && TexNativa::NegrasLinea(2)) {
				std::printf("[negra] vista-sin-ligar base=0x%llx origen=%s fmt=%u %ux%u niveles=%u vista=%u+%u mip=%u "
				            "nivel_fijado=%u\n",
				            static_cast<unsigned long long>(image.info.data.address), TexNativa::NegOrigen(image.neg_origen),
				            static_cast<uint32_t>(image.info.guest_format), image.info.extent.width,
				            image.info.extent.height, image.info.resources.levels, first, desc.view_info.level_count, level,
				            TexNativa::MinLevel(image.info.data.address));
				std::fflush(stdout);
			}
		}
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
	if (!image.usage.render_target) ++image.validity;
	image.usage.render_target = true;
	GestorTex::Reclasifica(image); // (M7) RT: solo en la GPU
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
	if (!image.usage.depth_target) ++image.validity;
	image.usage.depth_target = true;
	GestorTex::Reclasifica(image); // (M7) profundidad: solo en la GPU
	const bool scoped = RefreshDepthLayers(id, desc.view_info);
	if (!scoped) {
		RefreshImage(id);
	}
	if (desc.info.HasMetadata()) {
		// The epoch moves only when metadata state changes: every depth target
		// acquisition passes here.
		bool changed        = !(image.info.metadata == desc.info.metadata);
		if (changed) ++image.validity;
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
	CommitGpuWrite(image, scoped);
	NoteDepthLayersRendered(image, desc.view_info.base_layer, desc.view_info.layer_count);
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

void TextureCache::CommitGpuWrite(Image& image, bool keep_partial) {
	if (image.depth_id || image.backing.image == nullptr) {
		EXIT("TextureCache: stencil association cannot own image contents\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty() && !(keep_partial && image.IsPartiallyCpuDirty())) {
		FinishRefresh(image);
	}
	image.MarkGpuModified();
	NoteNativeGpuWrite(image);
	Spec::NoteGpuWrite(image.info.data.address, image.info.data.size);
}

void TextureCache::NoteNativeGpuWrite(const Image& image) {
	// (KYTY_NATIVE_IMAGE_DOWNLOAD=0: GpuResourceManager::NativeFlushGpuWrites downloads no image; none is noted.)
	static const bool downloads = [] {
		const char* v = std::getenv("KYTY_NATIVE_IMAGE_DOWNLOAD");
		return v == nullptr || v[0] != '0';
	}();
	if (!downloads || !PageManager::NativeWriteWatch() || image.info.data.Empty() || Spec::Current() != nullptr) return;
	m_native_written.insert(image.serial);
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
	if (image.info.resources.layers >= 6 && (image.info.IsDepth() || image.depth_array_alias)) {
		ShadowPoolTrace::Count(ShadowPoolTrace::Find(image.info.data.address, image.info.resources.layers,
		                                             image.info.extent.width, image.info.extent.height),
		                       &ShadowPoolTrace::Pool::clears);
	}
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
	ReleaseDownloadPages(address, size);
	InvalidateCpuAliases(address, size);
}

void TextureCache::InvalidateCpuWrittenPages(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	std::scoped_lock lock {m_lock};
	ReleaseDownloadPages(address, size);
	InvalidateCpuAliases(address, size, true);
}

// KYTY_NATIVE_MEMORY=1: the images GPU work wrote since the last call, downloaded to guest memory as the linear images'
// readback does (TryDownloadImage: recorded now, written back on the priority thread once the GPU is done; the caller
// finishes the GPU and waits for those write-backs). Images it refuses (depth with a stencil association, a download
// plan it has none for, one larger than the download buffer's half, GPU-written buffer bytes over it) keep what guest
// memory holds, as without native memory.
uint64_t TextureCache::NativeDownloadWritten(uint64_t* bytes, uint64_t* refused) {
	if (m_native_written.empty()) return 0;
	std::scoped_lock     lock {m_lock};
	std::vector<ImageId> ids;
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered && m_native_written.contains(image.serial)) ids.push_back(id);
	});
	m_native_written.clear();
	uint64_t done = 0;
	for (const auto id: ids) {
		auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered || !image->IsGpuModified() || image->info.data.Empty()) continue;
		const auto image_bytes = image->info.data.size;
		if (TryDownloadImage(id)) {
			++done;
			if (bytes != nullptr) *bytes += image_bytes;
		} else if (refused != nullptr) {
			++*refused;
		}
	}
	return done;
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
#if defined(_WIN32)
	// (KYTY_GPU_ONLY_OFFLOAD: the dropped guest bytes under it back first, as its read of the backing below would fault
	// for them; here, not inside that read.)
	if (g_offload_alive.load(std::memory_order_relaxed) && GpuOnlyOffloadActive() && CommandScheduler::ThreadRecorder() == nullptr &&
	    TextureOffload::Overlaps(range.address, range.size)) {
		RestoreOffloaded(range.address, range.size, static_cast<uint8_t>(TextureOffload::Touch::Readback));
	}
#endif
	auto&      utility = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	// (Larger than the download buffer, it could not be mapped: not downloaded, and kept by the caller.)
	// (W14) Salvo un RT grande viejo (g_rtg_descarga_propia): bufer de descarga propio del tamano exacto, que vive hasta
	// la escritura en la RAM del juego (lo guarda la operacion diferida).
	std::shared_ptr<Buffer> propio;
	if (range.size > utility.Size() / 2) {
		if (!g_rtg_descarga_propia || range.size > RtGrandeDescargaMax) return false;
		propio = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                  vk::BufferUsageFlagBits::eTransferDst,
		                                  range.size);
		if (propio->Mapped().size() < range.size) return false;
	}
	Buffer&  download = propio ? *propio : static_cast<Buffer&>(utility);
	uint8_t* mapped   = nullptr;
	uint64_t offset   = 0;
	if (propio) {
		mapped = propio->Mapped().data();
	} else {
		std::tie(mapped, offset) = utility.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
		if (mapped == nullptr) {
			EXIT("TextureCache: failed to map reusable download buffer\n");
		}
		utility.Commit();
	}
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
	// The pages stay write-watched until the write-back: a CPU write before it (the collector deletes the image at
	// once, a CPU write untracks it) is newer than the image's bytes, and takes its page from the write-back
	// (ReleaseDownloadPages). Written back whole, it overwrote such writes (game data in reused memory).
	const auto first_page = range.address & ~(TRACKER_PAGE_SIZE - 1);
	const auto end_page   = (range.address + range.size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	RtNativo::MarcaDescarga(range.address, range.size); // (RT-NATIVO2) ese rango de RAM ya no es la reserva virgen
	if (RtNativo::Es(image)) { // (RT-NATIVO) descarga pedida (expulsion, solape, lectura): se cuenta
		RtNativo::c_descargas.fetch_add(1, std::memory_order_relaxed);
		RtNativo::c_descargas_bytes.fetch_add(range.size, std::memory_order_relaxed);
	}
	m_page_manager.UpdatePageWatchers<true>(first_page, end_page - first_page);
	{
		std::lock_guard lock(m_download_mutex);
		m_pending_downloads.push_back(
		    {range.address, range.size, tick, first_page, std::vector<bool>((end_page - first_page) / TRACKER_PAGE_SIZE, true)});
	}
	m_scheduler.DeferPriorityOperation([this, &download, propio, range, mapped, offset, tick] {
		download.Invalidate(offset, range.size); // (W14) con bufer propio, `propio` lo mantiene vivo hasta aqui
		std::lock_guard lock(m_download_mutex);
		const auto done = std::ranges::find_if(m_pending_downloads, [&](const PendingDownload& pending) {
			return pending.address == range.address && pending.size == range.size && pending.tick == tick;
		});
		if (done == m_pending_downloads.end()) return;
		// The pages it still holds, by runs: written back, then let go.
		const auto& held = done->held;
		for (size_t first = 0; first < held.size();) {
			if (!held[first]) {
				++first;
				continue;
			}
			auto last = first + 1;
			while (last < held.size() && held[last]) ++last;
			const auto run_begin = done->first_page + first * TRACKER_PAGE_SIZE;
			const auto run_end   = done->first_page + last * TRACKER_PAGE_SIZE;
			const auto begin     = std::max(run_begin, range.address);
			const auto end       = std::min(run_end, range.address + range.size);
			LibKernel::Memory::WriteBacking(begin, mapped + (begin - range.address), end - begin, "image");
			m_page_manager.UpdatePageWatchers<false>(run_begin, run_end - run_begin);
			first = last;
		}
		m_pending_downloads.erase(done);
	});
	return true;
}

// The offload's downloads (checks, restores) into `host` (system memory, `total` bytes, zeros where no subresource lands):
// each image detiled into a video-memory scratch buffer, then one linear copy over. The tiler's scattered writes went
// straight over the bus before, a few bytes at a time: after each scan the frame's GPU work ended hundreds of ms late
// and the game waited for it (walk frames of 130-830 ms, the render thread idle). The scratch buffer lives until the GPU
// is done with it (the caller keeps it).
std::unique_ptr<Buffer> TextureCache::StageOffloadDownloads(std::vector<std::pair<Image*, DownloadPlan>>& images,
                                                            const std::vector<uint64_t>& offsets, uint64_t total,
                                                            Buffer& host) {
	auto scratch = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, total);
	m_scheduler.EndRendering();
	scratch->Fill(0, total, 0);
	for (size_t i = 0; i < images.size(); i++)
		DownloadImageData(*images[i].first, *scratch, offsets[i], images[i].first->info.data.size, std::move(images[i].second));
	m_scheduler.EndRendering();
	host.CopyFrom(m_scheduler.Current(), *scratch, 0, 0, total);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = host.Handle();
	barrier.offset              = 0;
	barrier.size                = total;
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost,
	                                               {}, 0, nullptr, 1, &barrier, 0, nullptr);
	return scratch;
}

// KYTY_TEXTURE_OFFLOAD(_PROBE) (texture-offload.h): every 120 flips, a sampled-only texture image clean at two scans in
// a row (GPU work never wrote it, no CPU write since its upload, tracked whole) has its whole pages probed, or dropped
// once a download of it gave its guest bytes back exactly (checked here first, 16 MiB of images a scan by default).
void TextureCache::OffloadScan(uint64_t tick) {
#if defined(_WIN32)
	// (Diagnostics only: the early exits that mark an image "differs" (m_offload_exact = false), counted per 30 s window.)
	static std::atomic<uint64_t> differs_depth {0}, differs_nobacking {0}, differs_plan {0}, differs_other {0};
	const bool offload = TextureOffload::GetMode() == TextureOffload::Mode::Offload;
	// (ram3, diagnostics) Why an image is in "differs", by serial (this thread only; erased with m_offload_exact's entry).
	enum DiffersWhy : uint8_t { WhyNone, WhyDepthTile, WhyKind, WhyUnmapped, WhyPlan, WhyCompare, WhyGone, WhyReasons };
	static std::unordered_map<uint64_t, uint8_t> differs_why;
	// (ram3) KYTY_OFFLOAD_PARTIAL=1 (off by default): a texture whose whole pages are only partly mapped - a streamed
	// texture held from its base level while the game has its largest mips unmapped (Demon's Souls: decommit at
	// eboot+0x9014f4) - was "differs-nobacking" for good: never checked, never taken. With it, its mapped runs are
	// compared with the download and taken run by run (a range each; a restore gives each its part of the image back).
	// The runs taken must be the runs checked: a mapping that came or went since has it checked again.
	static const bool partial_on = [] {
		const char* value = std::getenv("KYTY_OFFLOAD_PARTIAL");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			std::printf("Texture offload: partly mapped textures taken too, by their mapped runs (KYTY_OFFLOAD_PARTIAL=1)\n");
			std::fflush(stdout);
		}
		return on;
	}();
	static std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> partial_runs; // (serial -> runs checked)
	// (w6, KYTY_VRAM_UNICA) serial -> the pieces its check found equal to the download (the rest stays in RAM).
	static std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> unica_runs;
	// The mapped runs [first, second) of [from, to), neighbours merged.
	const auto mapped_runs = [](uint64_t from, uint64_t to, std::vector<std::pair<uint64_t, uint64_t>>& runs) {
		runs.clear();
		std::vector<std::pair<uint64_t, uint64_t>> parts;
		Libs::LibKernel::Memory::MappedParts(from, to - from, &parts);
		for (const auto& [address, bytes]: parts) {
			const uint64_t first = std::max(address, from), last = std::min(address + bytes, to);
			if (first >= last) continue;
			if (!runs.empty() && runs.back().second == first) runs.back().second = last;
			else runs.emplace_back(first, last);
		}
	};
	TextureOffload::Drain();
	TextureOffload::Report(false);
	// (w12, KYTY_VRAM_UNICA) Images whose different pieces were uploaded again from guest memory (once each), and the
	// ranges to upload again now. Guest memory is the truth for these textures (CPU-written, never GPU-written: the scan
	// only checks those); a different piece is a GPU copy left old (a streaming chunk written - Ghidra 0x901330 memcpy
	// 0x901459 - that its image never got). Uploaded again, the next check finds it equal and it leaves RAM too.
	static std::unordered_set<uint64_t>                  unica_reup;
	std::vector<std::pair<uint64_t, uint64_t>>           unica_reup_ranges;
	if (unica_reup.size() > 200000) unica_reup.clear();
	std::erase_if(m_offload_checks, [&](const std::unique_ptr<OffloadCheck>& check) {
		if (!check->done.load(std::memory_order_acquire)) return false;
		for (const auto& result: check->results) {
			if (VramUnica::On() && result.equal && result.known && unica_reup.erase(result.serial) != 0)
				VramUnica::C().fixed_after_reup.fetch_add(1, std::memory_order_relaxed);
			// (w12) First time different: its different pieces uploaded again, the image checked again once clean.
			if (!result.equal && result.known && !result.differ_runs.empty() && VramUnica::On() &&
			    unica_reup.insert(result.serial).second) {
				for (const auto& run: result.differ_runs) {
					unica_reup_ranges.push_back(run);
					VramUnica::C().bytes_reup.fetch_add(run.second - run.first, std::memory_order_relaxed);
				}
				VramUnica::C().images_reup.fetch_add(1, std::memory_order_relaxed);
				unica_runs.erase(result.serial);
				m_offload_exact.erase(result.serial);
				continue;
			}
			if (!result.equal && result.known && !result.differ_runs.empty() && VramUnica::On())
				VramUnica::C().still_differ.fetch_add(1, std::memory_order_relaxed);
			// (w6, KYTY_VRAM_UNICA) Some pieces equal, some not (a streaming chunk): the equal ones go, no veto of the
			// image's kind or format (the equal pieces proved the format).
			if (!result.equal && result.known && !result.equal_runs.empty() && VramUnica::On()) {
				m_offload_exact[result.serial] = true;
				unica_runs[result.serial]      = result.equal_runs;
				VramUnica::C().images_split.fetch_add(1, std::memory_order_relaxed);
				VramUnica::C().vetoes_avoided.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			unica_runs.erase(result.serial);
			m_offload_exact[result.serial] = result.equal;
			if (!result.equal) differs_other.fetch_add(1, std::memory_order_relaxed);
			if (!result.equal) differs_why[result.serial] = result.known ? WhyCompare : WhyGone;
			if (!result.equal && result.known) m_offload_differ_kinds.insert(result.kind);
			if (result.equal && result.known) m_offload_good_formats.insert(result.format);
			if (!result.equal && result.known) m_offload_bad_formats.insert(result.format);
		}
		for (const auto& entry: check->entries) m_offload_checking.erase(entry.serial);
		return true;
	});
	// (w12) Caller holds m_lock (RunGarbageCollector): as a CPU write of those bytes would (partial re-upload of the levels).
	for (const auto& [first, last]: unica_reup_ranges) {
		ReleaseDownloadPages(first, last - first);
		InvalidateCpuAliases(first, last - first);
	}
	std::erase_if(m_offload_restores,
	              [](const std::unique_ptr<OffloadRestore>& restore) { return restore->done.load(std::memory_order_acquire); });
	// (The collector's retirements just recorded theirs: on their way now, for whoever waits for them.)
	if (m_offload_unsubmitted) {
		m_offload_unsubmitted = false;
		m_scheduler.Flush();
	}
	if (tick % 120 != 0 && !GestorTex::forzar_scan) return; // (M7) forzada por el gestor con la RAM en critico
	// (ram18, KYTY_RAM18) The cold ranges of the game's memory the RAM18 thread found (no image), taken here where the
	// textures are (the one thread that adds ranges): their read watchers on now, before the RAM18 thread writes them to
	// the disk store and drops them (texture-offload.h, Ram18TakeQueued).
	if (TextureOffload::Ram18On()) {
		for (const auto& [cold_begin, cold_size]: TextureOffload::Ram18TakeQueued()) {
			AsyncUpload::DrainRange(cold_begin, cold_size);
			m_page_manager.UpdateReadWatchers<true>(cold_begin, cold_size);
		}
	}
	// (Diagnostics: where a scan's time goes, printed when it took 10 ms or more.)
	using Clock          = std::chrono::steady_clock;
	const auto scan_start = Clock::now();
	double     ms[8] {}; // images, drain, watch, discard, protect, buffer, record, total
	const auto since = [](Clock::time_point from) { return std::chrono::duration<double, std::milli>(Clock::now() - from).count(); };
	// KYTY_GPU_ONLY_OFFLOAD: GPU-written images taken too; under critical RAM pressure images from 64 KiB.
	// (ram15: not with native memory - GPU results there go through the lazy write-back's put-ins, raiz3 -.)
	const bool         gpu_only = offload && GpuOnlyOffloadActive() && !TextureOffload::Native();
	const uint64_t     MinSize  = gpu_only && RamGuard::PressureLevel() >= RamGuard::Critical ? uint64_t {64} << 10u
	                                                                                         : uint64_t {4} << 20u;
	// (picos2) 4 MiB, was 256 KiB: each restore of a dropped texture waits for the whole GPU (a download), 100-400 ms on a
	// 16 GB PC out of RAM - most of them 0.7-1.4 MiB textures streamed in again. Fewer, bigger ranges keep most of the saving.
	constexpr uint64_t MaxSize     = uint64_t {512} << 20u;
	// (KYTY_TEXTURE_OFFLOAD_CHECK_MB, default 16: the images a scan checks. Their downloads go with the frame's GPU work: 128
	// MiB a scan made walk frames of 30-55 ms every 2 s, the game waiting for its GPU work behind them.)
	static const uint64_t CheckBudget = [] {
		const char* value = std::getenv("KYTY_TEXTURE_OFFLOAD_CHECK_MB");
		return (value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : uint64_t {16}) << 20u;
	}();
	const auto         pages_of    = [](const Image& image) {
        return std::pair {(image.info.data.address + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1),
                          image.info.data.End() & ~(TRACKER_PAGE_SIZE - 1)};
	};
	std::unordered_map<uint64_t, uint64_t>        seen;
	std::vector<ImageId>                          take;
	std::vector<std::pair<ImageId, DownloadPlan>> check;
	uint64_t                                      check_bytes = 0;
	uint64_t                                      h1_wait_budget = 0; // (H1)
	// (ram15, KYTY_NATIVE_OFFLOAD_TAKE_MB, default 256: the textures a scan takes by their format's proof - each one's
	// read watchers on the render thread -, the rest at the next scans.)
	static const uint64_t TrustBudget = [] {
		const char* value = std::getenv("KYTY_NATIVE_OFFLOAD_TAKE_MB");
		return (value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : uint64_t {256}) << 20u;
	}();
	uint64_t trusted_scan = 0;
	// KYTY_GPU_ONLY_OFFLOAD: why a GPU-written image's guest bytes cannot go (0: they can). Its restore is the download a
	// CPU read of a GPU-written image gets anyway, proven for its format by a texture check; none for depth (no download),
	// linear images (the CPU reads them back), ones a CPU read or a GPU buffer write is pending over, dirty ones.
	enum GpuBlock : int { GpuOk, GpuDepth, GpuDirty, GpuReadback, GpuUnproven, GpuSmall, GpuUntracked, GpuTooBig, GpuBlocks };
	uint64_t   rt_ok_bytes = 0, rt_offloaded_bytes = 0, rt_all_bytes = 0; // (ram1, KYTY_DS_RT_OFFLOAD, breakdown scans)
	const auto gpu_block = [&](ImageId id, const Image& image) -> int {
		if (image.depth_id || image.usage.depth_target || image.info.tile_mode == Prospero::TileMode::kDepth || image.usage.video_out)
			return GpuDepth;
		if (image.IsCpuDirty() || image.IsBufferModified()) return GpuDirty;
		if (!image.IsTracked() || !image.untracked_holes.empty()) return GpuUntracked;
		const auto [begin, end] = pages_of(image);
		if (end <= begin || end - begin < MinSize) return GpuSmall;
		if (image.backing.image == nullptr || image.info.data.size > MaxSize) return GpuTooBig;
		if (!image.info.IsTiled() || m_download_images.contains(id) || HasPendingDownload(image.info.data.address, image.info.data.size) ||
		    m_buffer_cache.HasGpuDirtyBytes(image.info.data.address, image.info.data.size))
			return GpuReadback;
		const auto format = OffloadFormat(image);
		// (ram1, KYTY_DS_RT_OFFLOAD: inside the render-target arenas no format proof is asked.)
		const bool rt_arena = DsRtArena(image);
		if (!rt_arena && (m_offload_bad_formats.contains(format) || m_offload_differ_kinds.contains(OffloadKind(image)) ||
		                  (TextureOffload::GpuOnlySetting() != 2 && !m_offload_good_formats.contains(format))))
			return GpuUnproven;
		const auto plan = BuildDownload(image);
		if (!plan.valid || plan.depth_target) return GpuUnproven;
		return GpuOk;
	};
	std::unordered_set<uint64_t> gpu_take; // (serials of `take` the GPU wrote)
	// (Diagnostics, every 30 s: the guest bytes the GPU caches know of, by what could be dropped from RAM.)
	static auto breakdown_last = Clock::now() - std::chrono::seconds(20);
	// (picos4) Off unless KYTY_OFFLOAD_DIAG=1: its walk of the caches' buffers and images, on this thread with the lock
	// held, was a 110-180 ms frame every 20-30 s on a 16 GB PC.
	static const bool diag      = [] { const char* v = std::getenv("KYTY_OFFLOAD_DIAG"); return v != nullptr && *v == '1'; }();
	const bool  breakdown      = diag && Clock::now() - breakdown_last >= std::chrono::seconds(30);
	if (breakdown) breakdown_last = Clock::now();
	enum TexState : int { TexOffloaded, TexVerified, TexUnverified, TexWaiting, TexDiffers, TexDirty, TexSmall, TexUntracked,
		                  TexTooBig, TexTouched, TexStates };
	uint64_t gpu_bytes[GpuBlocks] {}, gpu_offloaded = 0, gpu_touched = 0, gpu_all = 0, gpu_count = 0;
	uint64_t tex_bytes[TexStates] {}, tex_all = 0, tex_count = 0, other_all = 0;
	uint64_t tex_differs_count = 0; // (Diagnostics: sampled textures in the differs state.)
	// (ram16, KYTY_RAM16 diagnostics) The images over the game's VirtualMemoryGPU zone (Ghidra 01.007.000: its base
	// 0x10_0000_0000, lStack_f8 in 0x80d9a0; the streamed textures, [memmap] "VirtualMemo"): what the cache covers of it.
	constexpr uint64_t VmZoneBegin = 0x1000000000ull, VmZoneEnd = 0x1400000000ull;
	// (ramlibre 11-23) KYTY_VM_TEX_OFFLOAD_KB=<n> (off by default; 0 off): with native memory, a sampled texture of the
	// streamed textures' zone (VirtualMemo: the game's CPU never reads it, see above) leaves RAM from n KiB, not from MinSize
	// (4 MiB, picos2): log 20261123-123158 had 0.71 GB of VirtualMemo textures in RAM with their copy in video memory and
	// 0.04 GB taken, the streamed textures being 0.7-1.4 MiB. Its cost: each touch is a download ("re-descargas").
	// (texnativo 11-23) KYTY_TEXNATIVO=1|0 (default on): every sampled texture of VirtualMemo, whatever its size, leaves
	// RAM once uploaded, as a PC game frees its copy after the upload. Ghidra 01.007.000 (CTexture::SetCurrentStreamingChunk
	// 0x901330): a chunk is committed fresh (0x7fdf70 op 0, 0x9013fa) and written whole by one memcpy (0x901459); a lower
	// chunk is decommitted (op 2, 0x9014f4: an unmap here, the range dropped with no download); the T# keeps its base
	// address and only its BASE_LEVEL/LAST_LEVEL change (0x901540): no CPU read and no partial rewrite of a texture's bytes.
	// Logs 085737 / 121729 / 123158: 0 re-descargas with the 4 MiB floor. KYTY_VM_TEX_OFFLOAD_KB=<n> still sets the floor.
	// (picos-ram 11-23) Its floor 1 MiB, not 1 page: log 20261123-130426 (floor 1 KiB) had 38 SLOW InitializeImage of
	// VirtualMemo textures in 161 s (40-80 ms each, all in the track phase, 1.8 s in all) and 0.49 GB out of RAM; log
	// 20261123-125652 (floor 1 MiB) none in 254 s with 0.25-0.36 GB out, the RAM available the same (1.32 vs 1.38 GB mean).
	static const uint64_t VmTexMin = [] {
		const char*    v  = std::getenv("KYTY_VM_TEX_OFFLOAD_KB");
		const char*    tn = std::getenv("KYTY_TEXNATIVO");
		const bool     on = tn != nullptr && tn[0] == '1'; // (revert 11-23 tarde) off by default: no log proves a gain
		const uint64_t kb = v != nullptr && v[0] != '\0' ? std::strtoull(v, nullptr, 10) : (on ? 1024 : 0);
		const uint64_t b  = kb == 0 ? 0 : std::max<uint64_t>(kb << 10u, TRACKER_PAGE_SIZE);
		if (b != 0) std::printf("[texnativo] texturas de VirtualMemo fuera de RAM tras subirlas desde %llu KiB (KYTY_TEXNATIVO=0 lo apaga; resto desde 4 MiB)\n",
		                        static_cast<unsigned long long>(b >> 10u));
		else std::printf("[texnativo] apagado (KYTY_TEXNATIVO=0): texturas de VirtualMemo desde 4 MiB como antes\n");
		return b;
	}();
	// (etapa2, 11-23, KYTY_TEX_SOLO_GPU; local/tex-anticipada.h) Una textura de VirtualMemo creada de antemano (etapa 1) y
	// usada por el draw con la misma imagen sale de RAM por esta misma ruta de offload nativa sea cual sea su tamano: la
	// comprobacion de subida, la proteccion y la re-descarga ante cualquier lectura (CPU, refresco, hash, re-subida) son las
	// de siempre. El resto: minimo de siempre (ruta alternativa, con RAM).
	// (confiar-gpu 11-24) KYTY_TEX_CONFIAR_GPU=1 (apagado por defecto; solo con KYTY_NATIVE_TEXTURE_OFFLOAD): una textura
	// muestreada y limpia de VirtualMemo sale de RAM sin comparar su descarga con la RAM, por su formato de confianza.
	// Desensamblado 01.007.000, CTexture::SetCurrentStreamingChunk 0x901330: tabla de trozos en this+0x70 (0x48 bytes:
	// +4 tamano, +8 desplazamiento) sobre la base this+0x98; subir de trozo compromete (0x809ec0 op 0, 0x9013fa) y copia
	// el trozo entero (memcpy 0x901459); bajar descompromete desde el trozo pedido hasta el final (op 2, 0x9014f4). El
	// juego nunca lee esos bytes con la CPU. Log 084052: las 15 "distintas" (fmt 173/181/182 tile 9 2048x2048 12 mips
	// 0x560000) difieren desde +0x20000, el borde del primer trozo: lo de despues no esta en streaming en la RAM y la
	// imagen de la GPU es la valida. Ademas el primer fallo marcaba la clase entera (m_offload_differ_kinds) como
	// "distinta" sin comprobar: por eso solo se miraban 13-17 imagenes en toda la sesion. Cualquier toque la re-descarga
	// de la GPU como siempre (fallo de pagina).
	static const bool ConfiarGpu = [] {
		const char* v  = std::getenv("KYTY_TEX_CONFIAR_GPU");
		const bool  on = v != nullptr && v[0] == '1' && TextureOffload::Native(); // (apagado: colgaba 11-24 09:01)
		std::printf("[confiar-gpu] KYTY_TEX_CONFIAR_GPU=%d: texturas de VirtualMemo desde 1 MiB fuera de RAM sin comparar "
		            "(su copia vive en la GPU; un toque la re-descarga)%s\n",
		            on ? 1 : 0, (v != nullptr && v[0] == '1' && !on) ? " [pedido pero sin KYTY_NATIVE_TEXTURE_OFFLOAD]" : "");
		std::fflush(stdout);
		return on;
	}();
	const auto confiar_vm = [&](uint64_t begin) { return ConfiarGpu && begin >= VmZoneBegin && begin < VmZoneEnd; };
	// (H1) definido aqui (antes dentro del recorrido): tex_min lo usa (H2).
	static const bool h1_tex_vm = [] {
		const char* v = std::getenv("KYTY_H1_TEX_VM");
		const bool  r = v != nullptr && v[0] == '1';
		std::printf("[h1-tex] KYTY_H1_TEX_VM=%d: texturas enteras de VirtualMemo de formato probado fuera de RAM sin descarga de "
		            "comparacion, desde 1 MiB (H2)\n", r ? 1 : 0);
		std::fflush(stdout);
		return r;
	}();
	const auto tex_min = [&](uint64_t begin) {
		if (confiar_vm(begin)) return std::min<uint64_t>(VmTexMin != 0 ? VmTexMin : MinSize, uint64_t {1} << 20u);
		// (H2, 11-25) Con H1, el suelo de VirtualMemo es 1 MiB y no 4 MiB (MinSize, picos2): log 20261125-215220 [vram-unica]
		// "pequena (<minimo)" 263 MiB, las texturas de streaming son de 0,7-4 MiB (2048x2048 12 mips 0x560000 = 5,4 MiB las
		// mayores) y casi ninguna llegaba a la comprobacion (4 en toda la sesion). 1 MiB y no menos: log 20261123-130426
		// (suelo 1 KiB) 38 InitializeImage lentas; log 20261123-125652 (suelo 1 MiB) ninguna (ver VmTexMin).
		if (h1_tex_vm && TextureOffload::Native() && begin >= VmZoneBegin && begin < VmZoneEnd)
			return std::min<uint64_t>(VmTexMin != 0 ? VmTexMin : MinSize, uint64_t {1} << 20u);
		if (TextureOffload::Native() && begin >= VmZoneBegin && begin < VmZoneEnd && TexAnt::SoloGpu() && TexAnt::IsUsed(begin))
			return static_cast<uint64_t>(TRACKER_PAGE_SIZE);
		return VmTexMin != 0 && TextureOffload::Native() && begin >= VmZoneBegin && begin < VmZoneEnd ? VmTexMin : MinSize;
	};
	uint64_t           vm_tex = 0, vm_tex_out = 0, vm_tex_count = 0, vm_gpu = 0, vm_other = 0;
	std::vector<std::pair<uint64_t, ImageId>> differs_list; // (ram3, diagnostics: their whole-page bytes and ids)
	const auto note = [&](ImageId id, const Image& image) {
		if (!image.registered) return;
		const auto [begin, end] = pages_of(image);
		const uint64_t bytes    = end > begin ? end - begin : 0;
		const int      status   = bytes != 0 ? TextureOffload::RangeStatus(begin, bytes, image.serial) : 0;
		const bool     vm_zone  = bytes != 0 && begin >= VmZoneBegin && begin < VmZoneEnd; // (ram16)
		if (GpuWrittenImage(image)) {
			if (vm_zone) vm_gpu += bytes;
			gpu_all += bytes;
			gpu_count++;
			const bool rt = DsRtArena(image);
			if (rt) rt_all_bytes += bytes;
			if (status == 1) {
				gpu_offloaded += bytes;
				if (rt) rt_offloaded_bytes += bytes;
			} else if (status == 2) {
				gpu_touched += bytes;
			} else {
				const int block = gpu_block(id, image);
				gpu_bytes[block] += bytes;
				if (rt && block == GpuOk) rt_ok_bytes += bytes;
			}
			return;
		}
		if (!image.usage.texture || image.usage.video_out) {
			if (vm_zone) vm_other += bytes;
			other_all += bytes;
			return;
		}
		if (vm_zone) { // (ram16)
			vm_tex += bytes;
			vm_tex_count++;
			if (status == 1) vm_tex_out += bytes;
		}
		tex_all += bytes;
		tex_count++;
		int state = TexUnverified;
		if (status == 1) state = TexOffloaded;
		else if (status == 2) state = TexTouched;
		else if (image.depth_id || image.IsCpuDirty() || image.IsBufferModified()) state = TexDirty;
		else if (!image.IsTracked() || !image.untracked_holes.empty()) state = TexUntracked;
		else if (bytes < MinSize) state = TexSmall;
		else if (image.backing.image == nullptr || image.info.data.size > MaxSize) state = TexTooBig;
		else if (!m_offload_clean.contains(image.serial)) state = TexWaiting;
		else if (const auto exact = m_offload_exact.find(image.serial); exact != m_offload_exact.end())
			state = exact->second ? TexVerified : TexDiffers;
		tex_bytes[state] += bytes;
		if (state == TexDiffers) tex_differs_count++;
		if (state == TexDiffers) differs_list.emplace_back(bytes, id);
	};
	// (u1) [vram-unica]: bytes de texturas muestreadas que no salen de RAM, por motivo (foto de esta pasada).
	uint64_t   vu_rej[VramUnica::RejN] {}, vu_tex_bytes = 0, vu_vm_bytes = 0;
	const auto vu_note = [&](const Image& image, int why) {
		if (!VramUnica::On() || !image.registered || !image.usage.texture || image.usage.video_out || GpuWrittenImage(image)) return;
		const auto [b, e] = pages_of(image);
		if (e > b) vu_rej[why] += e - b;
	};
	m_slot_images.ForEach([&](ImageId id, Image& image) {
		if (breakdown) note(id, image);
		if (VramUnica::On() && image.registered && image.usage.texture && !image.usage.video_out && !GpuWrittenImage(image)) {
			const auto [b, e] = pages_of(image);
			if (e > b) {
				vu_tex_bytes += e - b;
				if (b >= VmZoneBegin && b < VmZoneEnd) vu_vm_bytes += e - b;
			}
		}
		if (RtNativo::Es(image)) { // (RT-NATIVO) fuera del offload: sin vigilancia de lectura, su RAM sin tocar
			vu_note(image, VramUnica::RejRtNativo); // (H2)
			return;
		}
		// (KYTY_GPU_ONLY_OFFLOAD: a GPU-written image the texture filter below refuses anyway, clean at two scans in a row.)
		if (gpu_only && image.registered && GpuWrittenImage(image)) {
			if (gpu_block(id, image) != GpuOk) return;
			const auto [begin, end] = pages_of(image);
			const auto first        = m_offload_clean.find(image.serial);
			seen[image.serial]      = first != m_offload_clean.end() ? first->second : tick;
			if (first == m_offload_clean.end() || TextureOffload::Taken(begin, end - begin, image.serial)) return;
			gpu_take.insert(image.serial);
			take.push_back(id);
			return;
		}
		if (!image.registered || image.depth_id || image.IsGpuModified() || image.IsCpuDirty() || image.IsBufferModified() ||
		    !image.usage.texture || image.usage.storage || image.usage.render_target || image.usage.depth_target ||
		    image.usage.video_out || !image.IsTracked() || !image.untracked_holes.empty()) {
			vu_note(image, (!image.IsTracked() || !image.untracked_holes.empty()) ? VramUnica::RejSinVigilancia : VramUnica::RejSucia);
			return;
		}
		const auto [begin, end] = pages_of(image);
		if (end <= begin || end - begin < tex_min(begin)) { // (ramlibre: was MinSize)
			vu_note(image, VramUnica::RejPequena);
			return;
		}
		const auto first   = m_offload_clean.find(image.serial);
		seen[image.serial] = first != m_offload_clean.end() ? first->second : tick;
		if (first == m_offload_clean.end()) {
			vu_note(image, VramUnica::RejEspera);
			return;
		}
		if (TextureOffload::Taken(begin, end - begin, image.serial)) {
			// (H2) Antes sin contar. Grande: el tope de Taken (KYTY_OFFLOAD_MAX_IMAGE_MB, 32 por defecto; los arrays de
			// VirtualMemo de 88-352 MiB, que van por [rnt]); si no, un rango suyo ya fuera, tocada una vez o retenida (RAM17).
			const uint64_t cap = TextureOffload::MaxImageBytes();
			vu_note(image, cap != 0 && end - begin > cap ? VramUnica::RejGrande : VramUnica::RejYaFuera);
			return;
		}
		if (!offload) {
			vu_note(image, VramUnica::RejTomada); // (H2)
			take.push_back(id);
			return;
		}
		if (image.backing.image == nullptr || image.info.data.size > MaxSize ||
		    HasPendingDownload(image.info.data.address, image.info.data.size)) {
			vu_note(image, VramUnica::RejEspera);
			return;
		}
		if (const auto exact = m_offload_exact.find(image.serial); exact != m_offload_exact.end()) {
			if (exact->second) {
				vu_note(image, VramUnica::RejTomada); // (H2)
				take.push_back(id);
			} else {
				const auto why = differs_why.find(image.serial);
				const int  w   = why == differs_why.end() ? WhyCompare : why->second;
				vu_note(image, w == WhyUnmapped ? VramUnica::RejSinRespaldo
				               : w == WhyPlan   ? VramUnica::RejPlan
				               : (w == WhyDepthTile || w == WhyKind) ? VramUnica::RejClase
				                                                     : VramUnica::RejDistinta);
			}
			return;
		}
		// (ram15, KYTY_NATIVE_TEXTURE_OFFLOAD; KYTY_NATIVE_OFFLOAD_TRUST_FORMAT=0: off) A whole-mapped texture of a format
		// (guest format + tiling) a check proved exact, none found to differ, and of no kind that differed, is taken
		// without its own check, outside the check budget (16 MiB a scan verified ~8 MiB/s: minutes for the GBs in use).
		// Measured with the backing view (11-15 22:29, 22:40, 23:15): 5759 images checked, 0 differ.
		static const bool trust_formats = [] {
			const char* value = std::getenv("KYTY_NATIVE_OFFLOAD_TRUST_FORMAT");
			return TextureOffload::Native() && !(value != nullptr && value[0] == '0');
		}();
		// (w6, KYTY_VRAM_UNICA) A streamed texture (VirtualMemo) is never taken by its format alone: its chunks can
		// differ from its image, each is checked.
		const bool unica_vm       = VramUnica::On() && begin >= VmZoneBegin && begin < VmZoneEnd;
		// (H1, 11-25) KYTY_H1_TEX_VM=1 (apagado por defecto): una textura de VirtualMemo mapeada entera, de un formato que una
		// comprobacion ya demostro exacto (y sin clase ni formato que haya salido distinto), sale de RAM sin su descarga de
		// comparacion. Datos: logs 20261125-210818 / -211827 / -212851: comprobadas 13+6 texturas y 222 trozos, distintos 0;
		// [vram-unica] vistas 1655 MiB (1288 en VirtualMemo) y solo ~64 MiB/30 s comprobados (16 MiB por pasada,
		// KYTY_TEXTURE_OFFLOAD_CHECK_MB): el resto esperaba presupuesto sin contarse. Ghidra 01.007.000 (0x901330): la CPU
		// del juego no lee esos bytes; una textura entera ya no recibe subidas de trozo (solo descompromiso, op 2, sin
		// descarga). La toma vacia antes las subidas pendientes (AsyncUpload::DrainRange) y cualquier toque la re-descarga.
		const bool trusted_format = confiar_vm(begin) ||
		                            (trust_formats && (!unica_vm || h1_tex_vm) && m_offload_good_formats.contains(OffloadFormat(image)) &&
		                             !m_offload_bad_formats.contains(OffloadFormat(image)) &&
		                             !m_offload_differ_kinds.contains(OffloadKind(image)));
		if (m_offload_checking.contains(image.serial) || (!trusted_format && check_bytes >= CheckBudget)) {
			if (!m_offload_checking.contains(image.serial)) h1_wait_budget += end - begin; // (H1)
			vu_note(image, VramUnica::RejEspera);
			return;
		}
		// (A texture in depth tiling has no download: BuildTextureTransfer ends the emulator on one. Of a kind a check found
		// to differ: not checked again.)
		if (image.info.tile_mode == Prospero::TileMode::kDepth || (!confiar_vm(begin) && m_offload_differ_kinds.contains(OffloadKind(image)))) {
			differs_depth.fetch_add(1, std::memory_order_relaxed);
			m_offload_exact[image.serial] = false;
			differs_why[image.serial]     = image.info.tile_mode == Prospero::TileMode::kDepth ? WhyDepthTile : WhyKind;
			vu_note(image, VramUnica::RejClase); // (H2: la primera pasada tambien)
			return;
		}
		bool whole = true;
		for (uint64_t at = begin; at < end;) {
			const auto mapping_end = Libs::LibKernel::Memory::BackingMappingEnd(at);
			const auto piece       = mapping_end > at ? std::min(mapping_end, end) - at : 0;
			if (piece == 0 || Libs::LibKernel::Memory::TryGetBackingPointer(at, piece) == nullptr) {
				whole = false;
				break;
			}
			at += piece;
		}
		partial_runs.erase(image.serial);
		if (!whole) {
			// (ram3, KYTY_OFFLOAD_PARTIAL: its mapped runs, when they are worth it and each has backing bytes.)
			std::vector<std::pair<uint64_t, uint64_t>> runs;
			uint64_t                                   mapped = 0;
			if (partial_on) {
				mapped_runs(begin, end, runs);
				for (const auto& [run_first, run_last]: runs) {
					bool backed = true;
					for (uint64_t at = run_first; at < run_last && backed;) {
						const auto mapping_end = Libs::LibKernel::Memory::BackingMappingEnd(at);
						const auto piece       = mapping_end > at ? std::min(mapping_end, run_last) - at : 0;
						backed = piece != 0 && Libs::LibKernel::Memory::TryGetBackingPointer(at, piece) != nullptr;
						at += std::max<uint64_t>(piece, 1);
					}
					if (!backed) {
						mapped = 0;
						break;
					}
					mapped += run_last - run_first;
				}
			}
			if (mapped < MinSize) {
				differs_nobacking.fetch_add(1, std::memory_order_relaxed);
				m_offload_exact[image.serial] = false; // (no backing bytes for all of it: never taken)
				differs_why[image.serial]     = WhyUnmapped;
				vu_note(image, VramUnica::RejSinRespaldo); // (H2: la primera pasada tambien)
				return;
			}
			partial_runs[image.serial] = std::move(runs);
		}
		auto plan = BuildDownload(image);
		// (No buffer the GPU wrote over it: its guest bytes are the image's.)
		if (!plan.valid || plan.depth_target || m_buffer_cache.HasGpuDirtyBytes(image.info.data.address, image.info.data.size)) {
			differs_plan.fetch_add(1, std::memory_order_relaxed);
			m_offload_exact[image.serial] = false;
			differs_why[image.serial]     = WhyPlan;
			partial_runs.erase(image.serial);
			vu_note(image, VramUnica::RejPlan); // (H2: la primera pasada tambien)
			return;
		}
		if (trusted_format && whole) {
			if (trusted_scan >= TrustBudget) {
				vu_note(image, VramUnica::RejEspera); // (H2) tope de la pasada (KYTY_NATIVE_OFFLOAD_TAKE_MB): a la siguiente
				return;
			}
			vu_note(image, VramUnica::RejTomada); // (H2)
			trusted_scan += image.info.data.size;
			if (unica_vm) H1TexVmBytes().fetch_add(image.info.data.size, std::memory_order_relaxed); // (H1)
			m_offload_exact[image.serial] = true;
			TextureOffload::NoteTrusted(image.info.data.size);
			take.push_back(id);
			return;
		}
		if (check_bytes >= CheckBudget) {
			h1_wait_budget += end - begin; // (H1) esperando presupuesto de comprobacion (antes sin contar)
			vu_note(image, VramUnica::RejEspera); // (H2)
			return;
		}
		vu_note(image, VramUnica::RejEspera); // (H2) comprobandose desde esta pasada
		check_bytes += (image.info.data.size + 255) & ~uint64_t {255};
		check.emplace_back(id, std::move(plan));
	});
	m_offload_clean.swap(seen);
	std::erase_if(m_offload_exact, [&](const auto& item) { return !m_offload_clean.contains(item.first); });
	std::erase_if(differs_why, [&](const auto& item) { return !m_offload_clean.contains(item.first); });
	std::erase_if(partial_runs, [&](const auto& item) { return !m_offload_clean.contains(item.first); });
	std::erase_if(unica_runs, [&](const auto& item) { return !m_offload_clean.contains(item.first); });
	if (VramUnica::On()) { // (u1) foto de esta pasada para la linea [vram-unica]
		for (int r = 0; r < VramUnica::RejN; r++) VramUnica::C().rej[r].store(vu_rej[r], std::memory_order_relaxed);
		VramUnica::C().scanned_tex.store(vu_tex_bytes, std::memory_order_relaxed);
		VramUnica::C().scanned_vm.store(vu_vm_bytes, std::memory_order_relaxed);
	}
	VramUnica::Report();
	{ // (H1) cada 30 s: lo tomado por formato probado en VirtualMemo y lo que espera presupuesto de comprobacion
		static auto h1_last = std::chrono::steady_clock::now();
		if (std::chrono::steady_clock::now() - h1_last >= std::chrono::seconds(30)) {
			h1_last = std::chrono::steady_clock::now();
			std::printf("[h1-tex] 30 s: VirtualMemo fuera de RAM por formato probado (sin descarga) %.0f MiB en total; esperando "
			            "presupuesto de comprobacion ahora %.0f MiB\n",
			            static_cast<double>(H1TexVmBytes().load(std::memory_order_relaxed)) / 1048576.0,
			            static_cast<double>(h1_wait_budget) / 1048576.0);
			std::fflush(stdout);
		}
	}
	if (breakdown) {
		constexpr double MiB = 1024.0 * 1024.0;
		uint64_t buffer_total = 0, buffer_gpu = 0, buffer_gpu_full = 0, buffer_cpu = 0, buffer_count = 0;
		const bool buffers = GuestGpu::IsGpuThread();
		if (buffers) m_buffer_cache.GuestUseBreakdown(buffer_total, buffer_gpu, buffer_gpu_full, buffer_cpu, buffer_count);
		std::printf("Guest RAM by GPU cache (30 s, MiB; GPU-only offload %s%s): GPU-written images %.0f in %llu (offloaded %.0f, "
		            "could drop %.0f, depth %.0f, dirty %.0f, read back/linear %.0f, format unproven %.0f, small %.0f, untracked "
		            "%.0f, too big %.0f, touched once %.0f); sampled textures %.0f in %llu (offloaded %.0f, verified %.0f, not "
		            "verified yet %.0f, clean <1 scan %.0f, differs %.0f, dirty %.0f, small %.0f, untracked %.0f, too big %.0f, "
		            "touched once %.0f); other images %.0f\n",
		            gpu_only ? "on" : "off", gpu_only && MinSize < (uint64_t {256} << 10u) ? ", from 64 KiB: RAM critical" : "",
		            gpu_all / MiB, static_cast<unsigned long long>(gpu_count), gpu_offloaded / MiB, gpu_bytes[GpuOk] / MiB,
		            gpu_bytes[GpuDepth] / MiB, gpu_bytes[GpuDirty] / MiB, gpu_bytes[GpuReadback] / MiB, gpu_bytes[GpuUnproven] / MiB,
		            gpu_bytes[GpuSmall] / MiB, gpu_bytes[GpuUntracked] / MiB, gpu_bytes[GpuTooBig] / MiB, gpu_touched / MiB,
		            tex_all / MiB, static_cast<unsigned long long>(tex_count), tex_bytes[TexOffloaded] / MiB,
		            tex_bytes[TexVerified] / MiB, tex_bytes[TexUnverified] / MiB, tex_bytes[TexWaiting] / MiB,
		            tex_bytes[TexDiffers] / MiB, tex_bytes[TexDirty] / MiB, tex_bytes[TexSmall] / MiB, tex_bytes[TexUntracked] / MiB,
		            tex_bytes[TexTooBig] / MiB, tex_bytes[TexTouched] / MiB, other_all / MiB);
		std::printf("Guest RAM by GPU cache (30 s, MiB): buffers %.0f in %llu%s (a page GPU-written %.0f, all GPU-written %.0f: "
		            "could drop, not offloaded; a page CPU-written %.0f); could drop now: %.0f (GPU-written images %.0f, verified "
		            "textures %.0f, GPU-only buffers %.0f); the rest of the guest store committed (RAM line) is the game's own "
		            "data (images and buffers can overlap)\n",
		            buffer_total / MiB, static_cast<unsigned long long>(buffer_count), buffers ? "" : " (not on the GPU thread: not counted)",
		            buffer_gpu / MiB, buffer_gpu_full / MiB, buffer_cpu / MiB,
		            (gpu_bytes[GpuOk] + tex_bytes[TexVerified] + buffer_gpu_full) / MiB, gpu_bytes[GpuOk] / MiB,
		            tex_bytes[TexVerified] / MiB, buffer_gpu_full / MiB);
		// (ram16) KYTY_RAM16=1: how much of the streamed-texture zone the cache knows of. The zone's bytes in RAM are in
		// "[ram16-censo]" (VirtualMemo); what is in RAM there and in no image here has no copy in video memory.
		static const bool ram16 = [] {
			const char* v = std::getenv("KYTY_RAM16");
			return v != nullptr && v[0] == '1';
		}();
		if (ram16)
			std::printf("[ram16-texturas] (30 s, MiB) imagenes de la cache en la zona VirtualMemoryGPU (0x10_0000_0000-0x14_0000_0000): "
			            "texturas muestreadas %.0f en %llu (fuera de RAM %.0f), escritas por la GPU %.0f, otras %.0f; el resto de lo "
			            "que la zona tiene en RAM no esta en ninguna imagen\n",
			            vm_tex / MiB, static_cast<unsigned long long>(vm_tex_count), vm_tex_out / MiB, vm_gpu / MiB, vm_other / MiB);
		if (DsRtOffloadMode() != 0)
			std::printf("DS RT offload (30 s, KYTY_DS_RT_OFFLOAD=%d): GPU-written images in the render-target arenas %.0f MiB: "
			            "out of RAM %.0f, could drop now %.0f (no format check asked); the rest held by the other gates\n",
			            DsRtOffloadMode(), rt_all_bytes / MiB, rt_offloaded_bytes / MiB, rt_ok_bytes / MiB);
		{
			const unsigned long long n_images = tex_differs_count;
			const double             mib_each = n_images != 0 ? tex_bytes[TexDiffers] / MiB / static_cast<double>(n_images) : 0.0;
			std::printf("Offload diag (30 s): differs-depth %llu, differs-nobacking %llu, differs-plan %llu, differs-other %llu, "
			            "imagenes en differs %llu (%.1f MiB, media %.2f MiB)\n",
			            static_cast<unsigned long long>(differs_depth.exchange(0, std::memory_order_relaxed)),
			            static_cast<unsigned long long>(differs_nobacking.exchange(0, std::memory_order_relaxed)),
			            static_cast<unsigned long long>(differs_plan.exchange(0, std::memory_order_relaxed)),
			            static_cast<unsigned long long>(differs_other.exchange(0, std::memory_order_relaxed)), n_images,
			            tex_bytes[TexDiffers] / MiB, mib_each);
		}
		// (ram3, diagnostics) The largest images in "differs", one by one: what they are, why they are there, where the
		// game keeps them, how much of them is mapped and how much of that was ever written (FreshMemory's 64 KiB
		// granules: never-written memory costs no RAM, so "escrito" is the most they can hold in RAM). At most 8
		// images and 20000 granule lookups a line (a few hundred microseconds every 30 s).
		if (!differs_list.empty()) {
			std::sort(differs_list.begin(), differs_list.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
			static const char* const reasons[WhyReasons] = {"?", "tile-depth", "tipo-que-ya-fallo", "sin-mapear-entera",
			                                                "sin-plan-o-escrita-por-GPU", "comparacion", "rango-ocupado"};
			static const char* const arenas[3] = {"-", "RTShared", "RTPersist"};
			auto*       fresh = FreshMemory::Enabled() ? FreshMemory::Get() : nullptr;
			std::string line;
			uint64_t    all_bytes = 0, all_mapped = 0, all_written = 0, lookups = 0;
			size_t      shown = 0;
			std::vector<std::pair<uint64_t, uint64_t>> parts;
			for (const auto& [bytes, id]: differs_list) {
				if (shown >= 8 || lookups >= 20000) break;
				const auto& image       = m_slot_images[id];
				const auto [begin, end] = pages_of(image);
				if (end <= begin) continue;
				uint64_t mapped = 0, written = 0;
				Libs::LibKernel::Memory::MappedParts(begin, end - begin, &parts);
				for (const auto& [address, part_bytes]: parts) {
					const uint64_t first = std::max(address, begin), last = std::min(address + part_bytes, end);
					if (first >= last) continue;
					mapped += last - first;
					for (uint64_t at = first; at < last && lookups < 20000; lookups++) {
						const uint64_t stop    = std::min(last, (at | (FreshMemory::Granule - 1)) + 1);
						const auto*    backing = Libs::LibKernel::Memory::TryGetBackingPointer(at, stop - at);
						const uint64_t granule = backing != nullptr && fresh != nullptr
						                             ? FreshMemory::GranuleOf(*fresh, reinterpret_cast<uint64_t>(backing))
						                             : UINT64_MAX;
						// (Unknown counts as written.)
						if (granule == UINT64_MAX || FreshMemory::IsWritten(*fresh, granule)) written += stop - at;
						at = stop;
					}
				}
				const auto  why_found = differs_why.find(image.serial);
				const int   why       = why_found != differs_why.end() && why_found->second < WhyReasons ? why_found->second : 0;
				const int   arena     = Loader::DemonsSoulsMemMap::RtArenaOf(begin, end);
				const auto& info      = image.info;
				char        item[256];
				std::snprintf(item, sizeof(item),
				              "%s0x%llx+%.1fM fmt=%u tile=%u %ux%ux%u lv=%u ly=%u %s arena=%s mapeado=%.1fM escrito=%.1fM",
				              shown == 0 ? "" : "; ", static_cast<unsigned long long>(info.data.address), bytes / MiB,
				              static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode), info.extent.width,
				              info.extent.height, info.extent.depth, info.resources.levels, info.resources.layers, reasons[why],
				              arenas[arena >= 0 && arena <= 2 ? arena : 0], mapped / MiB, written / MiB);
				line += item;
				all_bytes += bytes;
				all_mapped += mapped;
				all_written += written;
				shown++;
			}
			std::printf("Offload differs (30 s): las %zu mayores de %zu suman %.0f MiB de rango: mapeado %.0f MiB, escrito alguna "
			            "vez %.0f MiB (lo maximo que pueden ocupar en RAM): %s\n",
			            shown, differs_list.size(), all_bytes / MiB, all_mapped / MiB, all_written / MiB, line.c_str());
		}
		std::fflush(stdout);
	}
	ms[0] = since(scan_start);
	const auto report = [&] {
		ms[7] = since(scan_start);
		if (ms[7] < 10.0) return;
		std::printf("Texture offload: scan %.1f ms (images %.1f; %zu taken: drain %.1f, watch %.1f, discard %.1f, protect %.1f; %zu "
		            "checked: buffer %.1f, record %.1f)\n",
		            ms[7], ms[0], take.size(), ms[1], ms[2], ms[3], ms[4], check.size(), ms[5], ms[6]);
	};
	std::vector<std::pair<uint64_t, uint64_t>> discards;
	for (const auto id: take) {
		auto& image             = m_slot_images[id];
		const auto [begin, end] = pages_of(image);
		std::vector<TextureOffload::Piece> pieces;
		uint64_t                           covered = 0;
		// (w6, KYTY_VRAM_UNICA) An image split by its check: only its equal pieces, each its range (still backed whole,
		// not taken by another image); the different pieces stay in RAM.
		if (const auto split = unica_runs.find(image.serial);
		    offload && split != unica_runs.end() && !gpu_take.contains(image.serial)) {
			for (const auto& [first, last]: split->second) {
				if (last - first < (uint64_t {64} << 10u) || TextureOffload::Taken(first, last - first, image.serial)) continue;
				pieces.clear();
				covered = 0;
				for (uint64_t at = first; at < last;) {
					const auto  mapping_end = Libs::LibKernel::Memory::BackingMappingEnd(at);
					const auto  piece       = mapping_end > at ? std::min(mapping_end, last) - at : 0;
					const auto* backing     = piece != 0 ? Libs::LibKernel::Memory::TryGetBackingPointer(at, piece) : nullptr;
					if (backing == nullptr) break;
					pieces.push_back({reinterpret_cast<uint64_t>(backing), piece});
					covered += piece;
					at += piece;
				}
				if (covered != last - first) continue;
				auto run_mark = Clock::now();
				AsyncUpload::DrainRange(first, last - first);
				ms[1] += since(run_mark);
				TextureOffload::Add(first, last - first, pieces, image.serial, (static_cast<uint64_t>(id.generation) << 32u) | id.index,
				                    true, false);
				run_mark = Clock::now();
				m_page_manager.UpdateReadWatchers<true>(first, last - first);
				ms[2] += since(run_mark);
				discards.emplace_back(first, last - first);
				VramUnica::C().runs_taken.fetch_add(1, std::memory_order_relaxed);
				VramUnica::C().bytes_taken.fetch_add(last - first, std::memory_order_relaxed);
			}
			// (Never again by the whole-range path: its different pieces must stay in RAM. Checked again only once it
			// stops being clean - m_offload_exact's entry goes with m_offload_clean's.)
			m_offload_exact[image.serial] = false;
			unica_runs.erase(split);
			continue;
		}
		for (uint64_t at = begin; at < end;) {
			const auto  mapping_end = Libs::LibKernel::Memory::BackingMappingEnd(at);
			const auto  piece       = mapping_end > at ? std::min(mapping_end, end) - at : 0;
			const auto* backing     = piece != 0 ? Libs::LibKernel::Memory::TryGetBackingPointer(at, piece) : nullptr;
			if (backing == nullptr) break;
			pieces.push_back({reinterpret_cast<uint64_t>(backing), piece});
			covered += piece;
			at += piece;
		}
		// (ram3, KYTY_OFFLOAD_PARTIAL: an image not all mapped goes by the mapped runs its check compared, a range each;
		// other runs now - a mapping came or went - have it checked again.)
		if (covered != end - begin && partial_on && offload && !gpu_take.contains(image.serial) &&
		    !TextureOffload::Taken(begin, end - begin, image.serial)) {
			std::vector<std::pair<uint64_t, uint64_t>> runs;
			mapped_runs(begin, end, runs);
			const auto checked = partial_runs.find(image.serial);
			if (checked == partial_runs.end() || checked->second != runs) {
				m_offload_exact.erase(image.serial);
				continue;
			}
			for (const auto& [first, last]: runs) {
				if (last - first < (uint64_t {64} << 10u)) continue;
				pieces.clear();
				covered = 0;
				for (uint64_t at = first; at < last;) {
					const auto  mapping_end = Libs::LibKernel::Memory::BackingMappingEnd(at);
					const auto  piece       = mapping_end > at ? std::min(mapping_end, last) - at : 0;
					const auto* backing     = piece != 0 ? Libs::LibKernel::Memory::TryGetBackingPointer(at, piece) : nullptr;
					if (backing == nullptr) break;
					pieces.push_back({reinterpret_cast<uint64_t>(backing), piece});
					covered += piece;
					at += piece;
				}
				if (covered != last - first) continue;
				auto run_mark = Clock::now();
				AsyncUpload::DrainRange(first, last - first);
				ms[1] += since(run_mark);
				TextureOffload::Add(first, last - first, pieces, image.serial, (static_cast<uint64_t>(id.generation) << 32u) | id.index,
				                    true, false);
				run_mark = Clock::now();
				m_page_manager.UpdateReadWatchers<true>(first, last - first);
				ms[2] += since(run_mark);
				discards.emplace_back(first, last - first);
			}
			continue;
		}
		// (Two images of one scan over the same pages, aliases of a format: the first one only. A range's pages are one
		// image's.)
		if (covered != end - begin || TextureOffload::Taken(begin, end - begin, image.serial)) continue;
		// (Pending upload jobs read its backing first. Known before its protection: a fault on its pages finds it; a
		// game thread's waits for the render thread, which drops its pages first.)
		auto mark = Clock::now();
		if (offload) AsyncUpload::DrainRange(begin, end - begin);
		ms[1] += since(mark);
		TextureOffload::Add(begin, end - begin, pieces, image.serial, (static_cast<uint64_t>(id.generation) << 32u) | id.index,
		                    offload, gpu_take.contains(image.serial));
		mark = Clock::now();
		m_page_manager.UpdateReadWatchers<true>(begin, end - begin);
		ms[2] += since(mark);
		if (offload) {
			if (VramUnica::On()) { // (w12: the [ram-unica] line counts whole textures too)
				VramUnica::C().runs_taken.fetch_add(1, std::memory_order_relaxed);
				VramUnica::C().bytes_taken.fetch_add(end - begin, std::memory_order_relaxed);
			}
			discards.emplace_back(begin, end - begin);
			continue;
		}
		mark = Clock::now();
		for (const auto& piece: pieces) (void)TextureOffload::Protect(piece.backing, piece.size, PAGE_NOACCESS);
		ms[4] += since(mark);
	}
	// The pages dropped by the worker thread (DiscardVirtualMemory took 10-90 ms of the render thread a scan): until then
	// they hold their bytes, and a restore meanwhile has nothing to wait for.
	if (!discards.empty()) {
		OffloadWorker::Post([discards = std::move(discards)] {
			// (KYTY_TEXTURE_OFFLOAD_TRIM=0: the guest view's pages left to Windows.)
			static const bool trim = [] {
				const char* value = std::getenv("KYTY_TEXTURE_OFFLOAD_TRIM");
				return value == nullptr || value[0] != '0';
			}();
			for (const auto& [begin, size]: discards)
				if (TextureOffload::DiscardRange(begin) && trim && !TextureOffload::Native())
					Libs::LibKernel::Memory::TrimGuestRange(begin, size);
		});
	}
	if (check.empty()) {
		report();
		return;
	}
	auto mark     = Clock::now();
	auto entry    = std::make_unique<OffloadCheck>();
	entry->buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, check_bytes);
	ms[5]         = since(mark);
	mark          = Clock::now();
	uint64_t                                      offset = 0;
	std::vector<std::pair<Image*, DownloadPlan>> downloads;
	std::vector<uint64_t>                         offsets;
	for (auto& [id, plan]: check) {
		auto& image             = m_slot_images[id];
		const auto [begin, end] = pages_of(image);
		const auto& info = image.info;
		char        what[160];
		std::snprintf(what, sizeof(what), "fmt=%u tile=%u %ux%ux%u levels=%u layers=%u size=0x%llx bgra16=%d",
		              static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode), info.extent.width,
		              info.extent.height, info.extent.depth, info.resources.levels, info.resources.layers,
		              static_cast<unsigned long long>(info.data.size), info.bgra16 ? 1 : 0);
		entry->entries.push_back({image.serial, info.data.address, begin, end, offset, OffloadKind(image), what});
		entry->entries.back().format = OffloadFormat(image);
		if (const auto runs = partial_runs.find(image.serial); runs != partial_runs.end()) entry->entries.back().runs = runs->second;
		m_offload_checking.insert(image.serial);
		// (w12) Its pending upload jobs recorded first: the download reads the image after them, not before.
		if (VramUnica::On()) AsyncUpload::DrainRange(begin, end - begin);
		downloads.emplace_back(&image, std::move(plan));
		offsets.push_back(offset);
		offset += (image.info.data.size + 255) & ~uint64_t {255};
	}
	// (Bytes between subresources: zeros, as a restore leaves them.)
	entry->scratch = StageOffloadDownloads(downloads, offsets, check_bytes, *entry->buffer);
	ms[6]          = since(mark);
	auto* raw = entry.get();
	m_offload_checks.push_back(std::move(entry));
	report();
	m_scheduler.DeferPriorityOperation([raw] { OffloadWorker::Post([raw] {
		if (!g_offload_alive.load(std::memory_order_acquire)) {
			raw->done.store(true, std::memory_order_release);
			return;
		}
		raw->buffer->Invalidate(0, raw->buffer->Size());
		const auto* mapped = raw->buffer->Mapped().data();
		uint64_t    bytes = 0, failed = 0;
		static std::atomic<uint32_t> reports {0};
		for (const auto& item: raw->entries) {
			// The image's whole pages against guest memory's backing bytes, a mapping at a time.
			bool     equal = true, known = true;
			uint64_t first = 0;
			// (w6, KYTY_VRAM_UNICA) Native memory: every piece compared on its own, the equal ones kept.
			const bool unica = VramUnica::On() && TextureOffload::Native();
			bool       unica_diff = false;
			std::vector<std::pair<uint64_t, uint64_t>> unica_equal, unica_differ;
			// (Not over another image's range taken since (a texture streamed in where an offloaded one was): its bytes
			// are gone, a read of them a miss and zeros, "differs" for the kind. Discards run on this thread: none during
			// the compare.)
			if (TextureOffload::Overlaps(item.begin, item.end - item.begin)) equal = known = false;
			const auto compare = [&](uint64_t from, uint64_t to) {
				for (uint64_t at = from; at < to && equal;) {
					const auto  mapping_end = Libs::LibKernel::Memory::BackingMappingEnd(at);
					const auto  piece       = mapping_end > at ? std::min(mapping_end, to) - at : 0;
					const auto* backing     = piece != 0 ? Libs::LibKernel::Memory::TryGetBackingPointer(at, piece) : nullptr;
					const auto* downloaded  = mapped + item.offset + (at - item.address);
					// (ram15, native memory: the guest pages themselves, read with a guarded copy a MiB at a time - a page
					// made no-access since the scan stops it: not known -.)
					thread_local std::vector<uint8_t> native_copy;
					if (backing != nullptr && TextureOffload::Native()) {
						native_copy.resize(std::max<size_t>(native_copy.size(), size_t {1} << 20u));
						bool same = true, read = true;
						for (uint64_t done = 0; done < piece && same && read;) {
							const uint64_t n = std::min<uint64_t>(piece - done, uint64_t {1} << 20u);
							read = Common::GuardedCopy(native_copy.data(), backing + done, n) == 0;
							same = read && std::memcmp(native_copy.data(), downloaded + done, n) == 0;
							done += n;
						}
						known = read;
						if (unica && read) {
							if (same) {
								if (!unica_equal.empty() && unica_equal.back().second == at) unica_equal.back().second = at + piece;
								else unica_equal.emplace_back(at, at + piece);
								VramUnica::C().pieces_equal.fetch_add(1, std::memory_order_relaxed);
								VramUnica::C().bytes_equal.fetch_add(piece, std::memory_order_relaxed);
							} else {
								if (!unica_diff) first = at;
								unica_diff = true;
								if (!unica_differ.empty() && unica_differ.back().second == at) unica_differ.back().second = at + piece;
								else unica_differ.emplace_back(at, at + piece);
								// (w12, diagnostics) The GPU copy of the piece all zeros: never uploaded.
								bool zeros = true;
								for (uint64_t k = 0; k + 8 <= piece && zeros; k += 8) {
									uint64_t v = 0;
									std::memcpy(&v, downloaded + k, 8);
									zeros = v == 0;
								}
								if (zeros) VramUnica::C().pieces_gpu_zero.fetch_add(1, std::memory_order_relaxed);
								VramUnica::C().pieces_differ.fetch_add(1, std::memory_order_relaxed);
								VramUnica::C().bytes_differ.fetch_add(piece, std::memory_order_relaxed);
							}
							at += std::max<uint64_t>(piece, 1);
							continue;
						}
						equal = read && same;
						if (!equal) first = at; // (not searched further: the log's offset is the piece's)
						at += std::max<uint64_t>(piece, 1);
						continue;
					}
					known = backing != nullptr;
					equal = known && std::memcmp(backing, downloaded, piece) == 0;
					if (!equal && known) {
						first = at;
						while (first < at + piece && backing[first - at] == downloaded[first - at]) ++first;
					}
					at += std::max<uint64_t>(piece, 1);
				}
			};
			// (ram3, KYTY_OFFLOAD_PARTIAL: only the runs mapped at the scan, when not all of it was.)
			if (item.runs.empty()) compare(item.begin, item.end);
			else
				for (const auto& [from, to]: item.runs) compare(from, to);
			if (unica && known && unica_diff) equal = false; // (w6: its equal pieces go alone, below)
			if (!equal && known && reports.fetch_add(1, std::memory_order_relaxed) < 24) {
				std::printf("Texture offload: download differs from the guest bytes at +0x%llx: %s\n",
				            static_cast<unsigned long long>(first - item.address), item.what.c_str());
			}
			bytes += item.end - item.begin;
			failed += equal || !known ? 0 : 1;
			raw->results.push_back({item.serial, item.kind, equal, known});
			raw->results.back().format = item.format;
			if (unica && known && unica_diff) {
				raw->results.back().equal_runs  = std::move(unica_equal);
				raw->results.back().differ_runs = std::move(unica_differ);
			}
		}
		TextureOffload::NoteCheck(raw->entries.size(), bytes, failed);
		raw->done.store(true, std::memory_order_release);
	}); });
#else
	(void)tick;
#endif
}

// Render thread (a speculation's recorder refuses): the discarded ranges over [vaddr, vaddr + size) downloaded back from
// their images in one submission, the render thread waiting for it; their read watchers then go.
// (ram17) Native offload: an upload that reads guest [vaddr, vaddr + size) without InitializeImage first (a level
// refresh, a partial or chunked upload: the upload worker reads it later, and cannot wait for this thread) gets the
// dropped bytes there back first, as InitializeImage does. Without it, a texture whose largest levels the game decommitted
// (its taken range split, a remnant left dropped) and streamed in again was read as dropped bytes by the worker: the 3
// "perdidas" of 11-16 11:01 (128 KiB remnants, "MISSED read window").
void TextureCache::RestoreBeforeUpload(uint64_t vaddr, uint64_t size) {
#if defined(_WIN32)
	if (!TextureOffload::Native() || size == 0) return;
	if (TextureOffload::Overlaps(vaddr, size)) RestoreOffloaded(vaddr, size, static_cast<uint8_t>(TextureOffload::Touch::Upload));
	if (TextureOffload::IsRestoring(vaddr, size)) {
		if (m_offload_unsubmitted) {
			m_offload_unsubmitted = false;
			m_scheduler.Flush();
		}
		TextureOffload::WaitRestores(vaddr, size);
		if (TextureOffload::Overlaps(vaddr, size)) RestoreOffloaded(vaddr, size, static_cast<uint8_t>(TextureOffload::Touch::Upload));
	}
#else
	(void)vaddr, (void)size;
#endif
}

void TextureCache::RestoreOffloaded(uint64_t vaddr, uint64_t size, uint8_t touch_value, bool async) {
#if defined(_WIN32)
	if (CommandScheduler::ThreadRecorder() != nullptr) throw CommandScheduler::RecorderRefusal {"texture offload restore"};
	const auto touch  = static_cast<TextureOffload::Touch>(touch_value);
	// (ram15, native memory) Ranges the worker is dropping or restoring: waited for (their GPU work submitted first);
	// then those whose bytes are still in RAM go without a download, their read watchers now.
	if (TextureOffload::Native()) {
		if (TextureOffload::IsRestoring(vaddr, size)) {
			if (m_offload_unsubmitted) {
				m_offload_unsubmitted = false;
				m_scheduler.Flush();
			}
			TextureOffload::WaitRestores(vaddr, size);
		}
		for (const auto& [begin, bytes]: TextureOffload::ReleaseIntactNative(vaddr, size, touch))
			m_page_manager.UpdateReadWatchers<false>(begin, bytes);
		// (ram17, KYTY_RAM17) Ranges with a copy on disk: an image going leaves them out of RAM (orphans, no download);
		// any other touch reads them back from the file here (no GPU work). BeginRestore leaves them out.
		if (TextureOffload::Ram17On()) {
			if (touch == TextureOffload::Touch::ImageGone) (void)TextureOffload::OrphanFileBacked(vaddr, size);
			else (void)TextureOffload::FileRestore(vaddr, size, touch, true);
		}
	}
	auto       ranges = TextureOffload::BeginRestore(vaddr, size, touch, touch == TextureOffload::Touch::Count, async);
	if (ranges.empty()) return;
	const auto                         start = std::chrono::steady_clock::now();
	std::unique_lock<TrackingSpinLock> lock(m_lock, std::defer_lock);
	if (!m_lock.OwnedByCurrentThread()) lock.lock();
	struct Job {
		size_t       range  = 0;
		Image*       image  = nullptr;
		DownloadPlan plan   = {};
		uint64_t     offset = 0;
	};
	std::vector<Job> jobs;
	uint64_t         total = 0, bytes = 0, lost = 0;
	for (size_t i = 0; i < ranges.size(); i++) {
		const auto& range = ranges[i];
		for (const auto& piece: range.pieces) (void)TextureOffload::Protect(piece.backing, piece.size, PAGE_READWRITE);
		auto* image = m_slot_images.try_get(ImageId {static_cast<uint32_t>(range.image), static_cast<uint32_t>(range.image >> 32u)});
		auto  plan  = image != nullptr && image->serial == range.serial && image->backing.image != nullptr ? BuildDownload(*image)
		                                                                                                 : DownloadPlan {};
		if (!plan.valid || plan.depth_target) {
			lost++;
			std::printf("Texture offload: LOST 0x%llx..0x%llx (its image gone or not downloadable: zeros)\n",
			            static_cast<unsigned long long>(range.begin), static_cast<unsigned long long>(range.end));
			continue;
		}
		jobs.push_back({i, image, std::move(plan), total});
		total += (image->info.data.size + 255) & ~uint64_t {255};
	}
	if (!jobs.empty()) {
		auto                                          owned    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, total);
		auto&                                         download = *owned;
		std::vector<std::pair<Image*, DownloadPlan>> downloads;
		std::vector<uint64_t>                         offsets;
		for (auto& job: jobs) {
			downloads.emplace_back(job.image, std::move(job.plan));
			offsets.push_back(job.offset);
		}
		auto scratch = StageOffloadDownloads(downloads, offsets, total, download);
		if (async) {
			auto restore     = std::make_unique<OffloadRestore>();
			restore->buffer  = std::move(owned);
			restore->scratch = std::move(scratch);
			restore->lost   = lost;
			restore->start  = start;
			for (const auto& job: jobs) {
				const auto& range = ranges[job.range];
				restore->copies.push_back({job.range, job.offset + (range.begin - job.image->info.data.address)});
				restore->bytes += range.end - range.begin;
			}
			TextureOffload::NoteRestoreKind(touch, restore->bytes);
			restore->ranges = std::move(ranges);
			auto* raw       = restore.get();
			m_offload_restores.push_back(std::move(restore));
			m_offload_unsubmitted = true;
			m_scheduler.DeferPriorityOperation([this, raw] { OffloadWorker::Post([this, raw] {
				if (!g_offload_alive.load(std::memory_order_acquire)) {
					raw->done.store(true, std::memory_order_release);
					return;
				}
				raw->buffer->Invalidate(0, raw->buffer->Size());
				const auto* mapped = raw->buffer->Mapped().data();
				for (const auto& copy: raw->copies) {
					const auto* source = mapped + copy.source;
					for (const auto& piece: raw->ranges[copy.range].pieces) {
						TextureOffload::WriteBack(piece.backing, source, piece.size); LibKernel::Memory::NativeForgetWrites(piece.backing, piece.size);
						source += piece.size;
					}
				}
				for (const auto& range: raw->ranges) m_page_manager.UpdateReadWatchers<false>(range.begin, range.end - range.begin);
				const auto us = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - raw->start).count());
				TextureOffload::EndRestore(raw->ranges, raw->bytes, us, raw->lost);
				raw->done.store(true, std::memory_order_release);
			}); });
			return;
		}
		m_scheduler.Wait(m_scheduler.CurrentTick());
		download.Invalidate(0, total);
		const auto* mapped = download.Mapped().data();
		for (const auto& job: jobs) {
			const auto& range  = ranges[job.range];
			const auto* source = mapped + job.offset + (range.begin - job.image->info.data.address);
			for (const auto& piece: range.pieces) {
				TextureOffload::WriteBack(piece.backing, source, piece.size); LibKernel::Memory::NativeForgetWrites(piece.backing, piece.size);
				source += piece.size;
			}
			bytes += range.end - range.begin;
		}
	}
	for (const auto& range: ranges) m_page_manager.UpdateReadWatchers<false>(range.begin, range.end - range.begin);
	// (ram17) A late restore, after a miss: whoever copied the range meanwhile (the upload worker into a buffer or an image)
	// copied dropped bytes. Its pages are made CPU-written for the next submission (KYTY_RAIZ's resync: the caches over
	// them invalidated and uploaded again from the bytes just restored), so no copy keeps them past it.
	if (touch == TextureOffload::Touch::Count && TextureOffload::Native() && Raiz::On())
		for (const auto& range: ranges) Raiz::QueueResync(range.begin & ~uint64_t {0xfff}, (range.end + 0xfff) & ~uint64_t {0xfff});
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	TextureOffload::EndRestore(ranges, bytes, us, lost);
	if (bytes != 0) TextureOffload::NoteRestoreKind(touch, bytes);
	static std::atomic<uint32_t> reports {0};
	if (us >= 2000 && reports.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("Texture offload: restore of %zu ranges (%.1f MiB, %s) took %.1f ms\n", ranges.size(), bytes / 1048576.0,
		            touch == TextureOffload::Touch::Count ? "after a miss" : TextureOffload::TouchName(touch), us / 1000.0);
	}
#else
	(void)vaddr, (void)size, (void)touch_value;
#endif
}

bool TextureCache::HasPendingDownload(uint64_t address, uint64_t size) {
	std::lock_guard lock(m_download_mutex);
	return std::ranges::any_of(m_pending_downloads, [&](const PendingDownload& pending) {
		return address < pending.address + pending.size && pending.address < address + size;
	});
}

void TextureCache::ReleaseDownloadPages(uint64_t address, uint64_t size) {
	std::lock_guard lock(m_download_mutex);
	for (auto& pending: m_pending_downloads) {
		if (address >= pending.address + pending.size || pending.address >= address + size) continue;
		const auto begin = std::max(address, pending.address) & ~(TRACKER_PAGE_SIZE - 1);
		const auto end   = std::min(address + size, pending.address + pending.size);
		for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
			const auto index = (page - pending.first_page) / TRACKER_PAGE_SIZE;
			if (index >= pending.held.size() || !pending.held[index]) continue;
			pending.held[index] = false;
			m_page_manager.UpdatePageWatchers<false>(page, TRACKER_PAGE_SIZE);
			LiveCounters::Add(LiveCounters::DownloadPagesReleased);
		}
	}
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

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size, const char* source, const uint32_t* fill) {
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
		// (The written bytes of it: InitializeImage uploads only the subresources over them.)
		const auto begin = std::max(address, image.info.data.address), end = std::min(address + size, image.info.data.End());
		if (fill != nullptr) {
			image.MarkBufferFilled(begin, end, *fill);
		} else {
			image.MarkBufferModified(begin, end);
		}
	}
}

bool TextureCache::HasTrackedDataOverlap(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) return true;
	// (m8) Fallo de lectura de una pagina vigilada en un hilo que YA tiene m_lock (InvalidateMemory / subida tocando
	// memoria del juego leida-protegida por la cache de buferes): sin copy feedback (true = solapa), la lectura sigue por
	// el camino normal del bufer. Antes: EXIT "recursive region tracking lock" (crash al ir al Nexo).
	if (m_lock.OwnedByCurrentThread()) return true;
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

// (etapa2, 11-23) See textureCache.h: KYTY_VRAM8_BC_BOUND is on by default on a small video memory heap.
bool TextureCache::SmallVramHeap() const {
	return m_graphics.small_vram_heap;
}

// (etapa 2.1, 11-23, KYTY_TEX_DIRECTA) See textureCache.h. Ghidra (r19g): cada trozo que copia CTexture::
// SetCurrentStreamingChunk (0x901330, memcpy en 0x901459) cae sobre mips/capas enteros de su textura (0x901030/0x901110
// reservan/liberan mip*caras*tamano_slice desde la base del T#); aqui ese rango se sube a la imagen por el plan de
// tiles del emulador (el mismo detiling que la subida normal). La RAM, la proteccion y el refresco normal no cambian.
// (etapa 2.1d) Sin hash ni subidas dobles. La 2.1c subia el trozo con UploadImageSubresources (RestoreBeforeUpload y
// ObtainBufferForImage de la imagen ENTERA, hasta 64 MiB, por cada trozo, y un hash de la RAM) y no marcaba nada como
// limpio, asi que el camino del draw lo volvia a subir: +2,3 ms de CPU por fotograma (logs 220332 vs 221742). Ahora es la
// misma subida parcial que RefreshTextureLevels: las paginas del trozo se vigilan ANTES de leer la RAM (una escritura
// despues vuelve a ensuciarlas), UploadImagePartial sube solo los subrecursos sucios dentro de esas paginas, y
// RefreshRangeComplete las da por limpias: el draw ya no las sube. expected_hash no se usa (la 2.1c valido 117 de 117).
int TextureCache::DirectUploadRange(uint64_t address, uint64_t size, uint64_t /*expected_hash*/, uint32_t* diff_fmt,
                                     uint32_t* diff_tile, uint64_t* diff_offset) {
	if (size == 0 || !GuestRange {address, size}.Valid()) return TexDirecta::kInvalid;
	std::scoped_lock lock {m_lock};
	const auto images = FindImagesInRegion(address, size, false);
	if (images.empty()) return TexDirecta::kNoImage;
	if (images.size() != 1) return TexDirecta::kOutside; // alias / vistas que se solapan: camino normal
	const auto id    = images.front();
	auto&      image = m_slot_images[id];
	const auto& info = image.info;
	if (diff_fmt != nullptr) *diff_fmt = static_cast<uint32_t>(info.guest_format);
	if (diff_tile != nullptr) *diff_tile = static_cast<uint32_t>(info.tile_mode);
	if (diff_offset != nullptr) *diff_offset = address - info.data.address;
	// Formato 181 / tile 9: los "difieren" de la 2.1/2.1c. Siempre por la ruta alternativa (el camino del draw).
	if (static_cast<uint32_t>(info.guest_format) == 181u && static_cast<uint32_t>(info.tile_mode) == 9u) {
		return TexDirecta::kFmtAlt;
	}
	if (!image.usage.texture || image.usage.storage || image.usage.render_target || image.usage.depth_target ||
	    image.usage.video_out || image.depth_id || info.IsDepth() || info.HasStencil() ||
	    info.metadata.kind != ImageMetadataKind::None || info.IsVolume() || image.IsGpuModified() ||
	    image.IsStencilModified() || image.IsBufferModified() || !image.registered) {
		return TexDirecta::kUnsafe;
	}
	if (address < info.data.address || address + size > info.data.End()) return TexDirecta::kOutside;
	if (info.data.size > (64ull << 20u)) return TexDirecta::kLarge;
	// Ya en la GPU: la imagen no tiene escrituras de la CPU pendientes en esas paginas.
	if (!image.IsCpuDirty()) return TexDirecta::kClean;
	if (image.IsMaybeCpuDirty() || kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) != 1 ||
	    !PartialDirtyCandidate(image) || info.data.address % TRACKER_PAGE_SIZE != 0) {
		return TexDirecta::kNotPartial;
	}
	// Paginas enteras del trozo, dentro de la imagen.
	const uint64_t begin = std::max(info.data.address, address & ~(static_cast<uint64_t>(TRACKER_PAGE_SIZE) - 1));
	const uint64_t end   = std::min(info.data.End(), (address + size + TRACKER_PAGE_SIZE - 1) & ~(static_cast<uint64_t>(TRACKER_PAGE_SIZE) - 1));
	const bool     whole = !image.IsPartiallyCpuDirty();
	if (whole ? image.IsTracked()
	          : image.track_addr != info.data.address || image.track_addr_end != info.data.End()) {
		return TexDirecta::kNotPartial;
	}
	if (!whole && !IntersectsRanges(image.CpuDirtyRanges(), begin, end)) return TexDirecta::kClean;
	// Vigiladas antes de leer los bytes (como RefreshTextureLevels): una escritura despues de la lectura da fallo.
	if (whole) {
		image.track_addr     = info.data.address;
		image.track_addr_end = info.data.End();
		WatchImagePages<true>(m_page_manager, image, begin, end - begin);
		if (info.data.address < begin) Image::AddRange(image.untracked_holes, info.data.address, begin);
		if (end < info.data.End()) Image::AddRange(image.untracked_holes, end, info.data.End());
		image.NarrowCpuDirty();
	} else {
		RetrackHoles(image, begin, end);
	}
	g_partial_fail = "";
	if (!UploadImagePartial(image, begin, end)) return TexDirecta::kUploadFailed; // sigue sucio: lo sube el draw
	image.staged_ring = nullptr;
	image.RefreshRangeComplete(begin, end);
	return TexDirecta::kOk;
}

void TextureCache::NativeGuardDirty(Image& image) {
	NativeApplyPending(image); // (11-25) los trozos de su rango que aun esperan en el anillo, antes de usarla
	// (AGUA segura, TEX-GUARDA) Imagen NO nativa (otro formato/forma/vista sobre la memoria de una CTexture nativa) que se
	// va a subir de la RAM: los bytes "solo en la GPU" no estan ahi (glitch de los charcos). La imagen nativa los devuelve
	// antes a la RAM (NativeRestoreBeforeDelete, sin borrarla; keep_on_fail: si no se puede, nada cambia). Seguro ahora:
	// solo se escriben rangos aun en el registro, sin cambio de epoca y con todas sus paginas protegidas contra escritura
	// de la CPU desde entonces (una reutilizacion por el juego hace fault y los saca antes). La 9f2fee5 escribia sin eso
	// (crash 20261126-220519). Una imagen ajena que escribe la GPU no la recibe (la RAM no es su origen).
	if (TexNativa::On() && image.IsCpuDirty() && !image.IsGpuModified() && !image.IsBufferModified() &&
	    !SparseIsPerLevel(image.backing) && TexNativa::AnyOwned(image.info.data.address, image.info.data.End())) {
		for (const auto other: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
			auto* o = m_slot_images.try_get(other);
			if (o == nullptr || o == &image || !o->registered || !SparseIsPerLevel(o->backing)) continue;
			if (NativeRestoreBeforeDelete(*o, true)) TexNativa::c_agua.fetch_add(1, std::memory_order_relaxed);
		}
	}
	// (NEGRAS-LOG a) Imagen no nativa sobre bytes "solo en la GPU": se sube de la RAM soltada (ceros).
	if (TexNativa::NegrasLog() && TexNativa::On() && image.IsCpuDirty() && !SparseIsPerLevel(image.backing) &&
	    TexNativa::AnyOwned(image.info.data.address, image.info.data.End())) {
		static std::unordered_set<uint64_t> seen;
		if (seen.size() < 100000 && seen.insert(image.serial).second && TexNativa::NegrasLinea(0)) {
			std::vector<std::pair<uint64_t, uint64_t>> owned;
			TexNativa::OwnedIn(image.info.data.address, image.info.data.End(), owned);
			uint64_t ob = 0;
			for (const auto& [b, e]: owned) ob += e - b;
			std::printf("[negra] ajena addr=0x%llx size=0x%llx origen=%s fmt=%u vk=%u %ux%u niveles=%u capas=%u tile=%u "
			            "disperso=%d buffer_mod=%d solo_gpu=%llu KB; nativas en su memoria:",
			            static_cast<unsigned long long>(image.info.data.address),
			            static_cast<unsigned long long>(image.info.data.size), TexNativa::NegOrigen(image.neg_origen),
			            static_cast<uint32_t>(image.info.guest_format), static_cast<uint32_t>(image.info.pixel_format),
			            image.info.extent.width, image.info.extent.height, image.info.resources.levels,
			            image.info.resources.layers, static_cast<uint32_t>(image.info.tile_mode), image.IsSparse() ? 1 : 0,
			            image.IsBufferModified() ? 1 : 0, static_cast<unsigned long long>(ob >> 10u));
			for (const auto other: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
				const auto* o = m_slot_images.try_get(other);
				if (o == nullptr || !SparseIsPerLevel(o->backing)) continue;
				std::printf(" {base_ctexture=0x%llx origen=%s fmt=%u vk=%u %ux%u niveles=%u size=0x%llx}",
				            static_cast<unsigned long long>(o->info.data.address), TexNativa::NegOrigen(o->neg_origen),
				            static_cast<uint32_t>(o->info.guest_format), static_cast<uint32_t>(o->info.pixel_format),
				            o->info.extent.width, o->info.extent.height, o->info.resources.levels,
				            static_cast<unsigned long long>(o->info.data.size));
			}
			std::printf("\n");
			std::fflush(stdout);
		}
	}
	if (!TexNativa::On() || !image.IsCpuDirty() || image.IsBufferModified() || !SparseIsPerLevel(image.backing)) return;
	const uint64_t base = image.info.data.address, end = image.info.data.End();
	std::vector<std::pair<uint64_t, uint64_t>> owned;
	TexNativa::OwnedIn(base, end, owned);
	if (owned.empty()) return;
	std::vector<std::pair<uint64_t, uint64_t>> want;
	if (image.IsPartiallyCpuDirty() && !image.IsMaybeCpuDirty()) want = image.CpuDirtyRanges();
	else want.assign(1, {base, end});
	for (const auto& [b, e]: owned) Image::SubtractRange(want, b, e);
	if (want.empty()) {
		// todo lo sucio era "solo en la GPU": ya esta en su textura
		if (image.IsTracked()) FinishRefresh(image);
		else image.RefreshComplete();
		return;
	}
	image.RefreshComplete();
	for (const auto& [b, e]: want) image.InvalidateCpuWritePartial(b, e);
}

// (N4) RAM real: un trozo ya en su textura y "solo en la GPU" no necesita sus paginas de la RAM emulada. MEM_RESET:
// Windows las da por libres sin escribirlas al archivo de paginacion (el contenido se descarta); despues, fuera del
// working set en el hilo de offload. Bajo PMutex: NativeChunkCopy no puede dejar pasar entretanto una escritura del
// juego a la RAM (la quita de "solo en la GPU" antes). La restauracion corre en este mismo hilo. KYTY_TEX_NATIVA_SOLTAR=0 la apaga.
static void NativeDropRam(uint64_t address, uint64_t end) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// (N5) KYTY_TEX_NATIVA_LIBERAR (antes KYTY_TEX_NATIVA_SOLTAR), encendido por defecto con KYTY_TEX_NATIVA=1.
	// El intento anterior (log 035931) solo sacaba las paginas del working set de la VISTA DE RESPALDO: la pagina fisica
	// de la seccion seguia en el working set de la vista del juego (la direccion invitada es la misma direccion del
	// host, GuestBackingStore::MapExistingPlaceholderFixed), por eso la RAM no bajo. Ahora: (1) solo paginas enteras,
	// del todo "solo en la GPU"; (2) nunca un rango cuya memoria fisica este mapeada tambien en otra direccion invitada
	// (alias del PS5: ahi el juego si puede leer datos vivos; hipotesis del cuelgue en eboot+0x80315f, sin verificar);
	// (3) MEM_RESET por la vista de respaldo y fuera del working set de las dos vistas.
	// (tex2) Dos condiciones nuevas: (a) el gancho de free del monton del juego (0x802f90) instalado: el juego reutiliza
	// los bloques liberados para los nodos de su arbol libre (cabecera en ptr-8); el gancho quita el bloque de "solo en
	// la GPU" ANTES de que escriba, y bajo PMutex (el mismo cerrojo de aqui: un reset nunca cae despues de su escritura).
	// Sin el gancho, el reset borraba esos nodos: cuelgue en eboot+0x802cb0 / +0x80315f leyendo 0xfff...f8 (log 042242).
	// (b) Memoria nativa (KYTY_NATIVE_MEMORY=1): MEM_RESET + VirtualUnlock sobre la memoria privada del juego, la misma
	// direccion que el juego usa (NativeOffloadDrop). Con la seccion del archivo de paginacion (dos vistas) la RAM no
	// bajaba: working set 6,0-8,4 GB igual que sin liberar; esa via queda solo con KYTY_TEX_NATIVA_LIBERAR=1.
	const bool native_memory = LibKernel::Memory::NativeMemoryEnabled();
	if (!TexNativa::Liberar(native_memory) || !TexNativa::g_free_hook_ok.load(std::memory_order_acquire)) return;
	const uint64_t b = (address + 0xfffu) & ~uint64_t {0xfff}, e = end & ~uint64_t {0xfff};
	if (e <= b) return;
	std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	std::lock_guard<std::mutex> lock(TexNativa::PMutex());
	std::vector<std::pair<uint64_t, uint64_t>> owned;
	TexNativa::OwnedIn(b, e, owned);
	uint64_t covered = 0;
	for (const auto& [ob, oe]: owned) covered += std::min(oe, e) - std::max(ob, b);
	if (covered != e - b) return; // solo paginas enteras sin dueno en la RAM
	if (native_memory) {
		// Privada, una sola direccion (memoria nativa: sin alias). Reset + fuera del working set, un MiB cada vez.
		uint64_t refused = 0;
		const uint64_t done = LibKernel::Memory::NativeOffloadDrop(b, e - b, &refused);
		TexNativa::c_soltado_bytes.fetch_add(done, std::memory_order_relaxed);
		TexNativa::c_soltado_fallo.fetch_add(refused, std::memory_order_relaxed);
		// (11-25d) Lo que NativeOffloadDrop no toco ni rechazo no estaba comprometido (KYTY_COMMIT_REAL: se compromete al
		// tocar, y el trozo directo no toca la RAM): sin respaldo ya, como lo soltado. Log 075201: "soltada 0, rechazadas 0".
		const uint64_t never = (e - b) - std::min(e - b, done + refused);
		TexNativa::c_sin_commit_bytes.fetch_add(never, std::memory_order_relaxed);
		if (refused != 0) return; // (lo no reseteado sigue respaldado: no cuenta como liberado)
		TexNativa::AddReleased(b, e);
		TexNativa::c_liberada_bytes.fetch_add(done, std::memory_order_relaxed); // (11-25d) solo lo reseteado de verdad
		return;
	}
	if (!LibKernel::Memory::IsUniqueGuestBackingRangeAcrossMappings(b, e - b)) {
		TexNativa::c_liberada_alias.fetch_add(e - b, std::memory_order_relaxed);
		return;
	}
	if (!LibKernel::Memory::TryGetBackingPieces(b, e - b, pieces)) return;
	uint64_t done = 0;
	for (const auto& [ptr, n]: pieces) {
		const uint64_t pb = (reinterpret_cast<uint64_t>(ptr) + 0xfffu) & ~uint64_t {0xfff};
		const uint64_t pe = (reinterpret_cast<uint64_t>(ptr) + n) & ~uint64_t {0xfff};
		if (pe <= pb) continue;
		if (VirtualAlloc(reinterpret_cast<void*>(pb), pe - pb, MEM_RESET, PAGE_READWRITE) == nullptr) {
			TexNativa::c_soltado_fallo.fetch_add(pe - pb, std::memory_order_relaxed);
			continue;
		}
		done += pe - pb;
		TexNativa::c_soltado_bytes.fetch_add(pe - pb, std::memory_order_relaxed);
		OffloadWorker::Post([pb, pe] {
			constexpr uint64_t Chunk = uint64_t {1} << 20u;
			for (uint64_t at = pb; at < pe; at += Chunk) {
				static thread_local uint64_t t_at = 0, t_size = 0;
				t_at   = at;
				t_size = std::min(Chunk, pe - at);
				LocalPlatform::RunTrimQuiet([] { (void)VirtualUnlock(reinterpret_cast<void*>(t_at), t_size); });
			}
		});
	}
	if (done != e - b) return; // (sin la vista del juego: lo no reseteado sigue respaldado)
	TexNativa::AddReleased(b, e);
	TexNativa::c_liberada_bytes.fetch_add(e - b, std::memory_order_relaxed);
	// La vista del juego (misma direccion que la invitada): fuera de su working set tambien. Sin cambiar su proteccion
	// (la del gestor de paginas). Si para entonces el rango ya no esta mapeado, VirtualUnlock falla sin efecto.
	OffloadWorker::Post([b, e] {
		constexpr uint64_t Chunk = uint64_t {1} << 20u;
		for (uint64_t at = b; at < e; at += Chunk) {
			static thread_local uint64_t t_at = 0, t_size = 0;
			t_at   = at;
			t_size = std::min(Chunk, e - at);
			LocalPlatform::RunTrimQuiet([] { (void)VirtualUnlock(reinterpret_cast<void*>(t_at), t_size); });
		}
	});
#else
	(void)address, (void)end;
#endif
}

bool TextureCache::NativeUploadHost(uint64_t address, uint64_t size, uint64_t ring_offset, uint64_t seq, uint64_t base) {
	std::scoped_lock lock {m_lock};
	return NativeUploadHostLocked(address, size, ring_offset, seq, nullptr, base);
}

// (11-25) Antes de usar una imagen nativa (InitializeImage / RefreshTextureLevels -> NativeGuardDirty): los trozos que
// esperan en el anillo para su rango se suben YA a sus mips, asi el draw nunca la usa sin ellos (y nunca se leen de la RAM,
// donde no estan). Sin la imagen registrada se dejan en la cola (los sube el drenaje). Se para en el primer aviso kRamWrite
// del rango: el orden de los trozos de un mismo rango se respeta.
void TextureCache::NativeApplyPending(Image& image) {
	if (!TexNativa::On() || !image.registered || !SparseIsPerLevel(image.backing)) return;
	const uint64_t base = image.info.data.address, end = image.info.data.End();
	if (!TexNativa::AnyOwned(base, end)) return;
	std::vector<TexNativa::Pending> mine;
	{
		std::lock_guard<std::mutex> plock(TexNativa::PMutex());
		auto&                       q = TexNativa::PQueue();
		for (auto it = q.begin(); it != q.end();) {
			if (it->dest + it->size <= base || it->dest >= end) {
				++it;
				continue;
			}
			if (it->ring_offset == TexNativa::kRamWrite || it->dest < base || it->dest + it->size > end) break;
			mine.push_back(*it);
			it = q.erase(it);
		}
	}
	for (const auto& p: mine) {
		(void)NativeUploadHostLocked(p.dest, p.size, p.ring_offset, p.seq, &image);
	}
}

bool TextureCache::NativeUploadHostLocked(uint64_t address, uint64_t size, uint64_t ring_offset, uint64_t seq, Image* known,
                                          uint64_t base_hint) {
	const uint64_t   end  = address + size;
	// (N2+N3d) Aviso: el juego escribio este trozo en la RAM (anillo lleno): la imagen, sucia ahi; nada que subir aqui.
	if (ring_offset == TexNativa::kRamWrite) {
		(void)TexNativa::ForgetReleased(address, end, true); // (N5) el juego escribio sus bytes: pagina viva otra vez
		TexNativa::RemoveOwned(address, end);
		for (const auto id: FindImagesInRegion(address, size, false)) {
			auto* owner = m_slot_images.try_get(id);
			if (owner == nullptr) continue;
			if (TryInvalidatePartial(*owner, address, size, TRACKER_PAGE_SIZE)) continue;
			owner->InvalidateCpuWrite(address, size);
			UntrackImage(id, "tex-nativa");
		}
		TexNativa::c_fallback_bytes.fetch_add(size, std::memory_order_relaxed);
		return false;
	}
	const uint8_t*   data = TexNativa::TheRing().base.load(std::memory_order_acquire) + ring_offset;
	// Sin subida directa posible: a la RAM emulada como siempre, y la imagen que haya ahi, sucia en ese rango.
	const auto fallback = [&]() {
		(void)TexNativa::ForgetReleased(address, end, true); // (N5)
		TexNativa::RemoveOwned(address, end);
		LibKernel::Memory::TryWriteBacking(address, data, size);
		for (const auto id: FindImagesInRegion(address, size, false)) {
			auto* owner = m_slot_images.try_get(id);
			if (owner == nullptr) continue;
			if (TryInvalidatePartial(*owner, address, size, TRACKER_PAGE_SIZE)) continue;
			owner->InvalidateCpuWrite(address, size);
			UntrackImage(id, "tex-nativa");
		}
		TexNativa::c_fallback_bytes.fetch_add(size, std::memory_order_relaxed);
		TexNativa::RingDone(seq, 0);
		return false;
	};
	if (size == 0 || !GuestRange {address, size}.Valid()) return fallback();
	Image* found = known;
	// (11-25c) La imagen nativa de su CTexture, directa por la base del T# (sin FindImagesInRegion; y aunque otra imagen
	// solape el rango, el trozo va a la de su CTexture).
	if (found == nullptr && base_hint != 0) {
		if (auto it = NativeByBase().find(base_hint); it != NativeByBase().end()) {
			for (auto k = it->second.rbegin(); k != it->second.rend(); ++k) { // la mas reciente primero
				auto* img = m_slot_images.try_get(*k);
				if (img != nullptr && img->registered && address >= base_hint && end <= img->info.data.End()) {
					found = img;
					TexNativa::c_by_base_hit.fetch_add(1, std::memory_order_relaxed);
					break;
				}
			}
		}
	}
	if (found == nullptr) {
		const auto images = FindImagesInRegion(address, size, false);
		if (images.size() != 1) return fallback();
		found = &m_slot_images[images.front()];
	}
	auto&       image = *found;
	const auto& info  = image.info;
	const auto  base  = info.data.address;
	if (!SparseIsPerLevel(image.backing) || !image.registered || image.IsGpuModified() || image.IsBufferModified() ||
	    image.IsStencilModified() || address < base || end > info.data.End() || info.IsVolume() || base % 256 != 0) {
		return fallback();
	}
	const auto binding = UploadBinding(image);
	if (binding == BindingType::DepthTarget) return fallback();
	auto& cached = m_partial_plans[image.serial];
	if (cached.plan == nullptr || cached.binding != binding) {
		cached.plan    = std::make_shared<TextureTransferPlan>(BuildTextureTransfer(image, binding, TransferDirection::Upload));
		cached.binding = binding;
	}
	const auto& plan = *cached.plan;
	if (!plan.valid || plan.swap_bgra16 || plan.tiles.empty() || plan.tiles.size() != plan.regions.size()) return fallback();
	// Los subrecursos (tiles) dentro del trozo, enteros; su origen, los bytes del trozo.
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         linear_total = 0;
	for (size_t index = 0; index < plan.tiles.size(); index++) {
		auto           tile   = plan.tiles[index];
		auto           region = plan.regions[index];
		const uint64_t tb     = base + tile.tiled_offset, te = tb + tile.tiled_size;
		if (te <= address || tb >= end) continue;
		if (tb < address || te > end || region.bufferOffset < tile.linear_offset) return fallback(); // tile partido
		const auto linear_offset = (linear_total + 255) & ~uint64_t {255};
		region.bufferOffset      = linear_offset + (region.bufferOffset - tile.linear_offset);
		tile.linear_offset       = linear_offset;
		tile.tiled_offset        = tb - address;
		linear_total             = linear_offset + tile.linear_size;
		tiles.push_back(tile);
		regions.push_back(region);
	}
	if (tiles.empty()) return fallback();
	// (NEGRAS) Un mip sin memoria ligada (por debajo de MinLevel, desligado o sin memoria): la subida se perderia y el
	// rango quedaria "solo en la GPU" con su RAM soltada = negro para siempre. A la RAM como siempre.
	// (MIP0-NATIVA) Antes de ir a la RAM: si el juego ya comprometio ese rango, se liga ahora (hilo de la GPU; el ligado
	// queda pendiente y SparseFlushPendingBinds lo envia antes de la entrega que lleva esta subida) y se vuelve a mirar.
	// Solo lo que de verdad no esta comprometido (o sin memoria de video) cae a la RAM.
	bool tried_bind = false;
	for (const auto& region: regions)
		if (!SparseLevelBound(image.backing, region.imageSubresource.mipLevel)) {
			if (!tried_bind && GuestGpu::IsGpuThread()) {
				tried_bind = true;
				image.SparseUpdate(address, end);
				if (SparseLevelBound(image.backing, region.imageSubresource.mipLevel)) continue;
			}
			TexNativa::c_unbound_fallback.fetch_add(size, std::memory_order_relaxed);
			return fallback();
		}
	if (m_native_ring == nullptr) return fallback();
	// Los bytes ya estan en la memoria de subida (los copio el juego): destilado directo desde ahi.
	m_native_ring->Flush(ring_offset, size);
	const auto linear = m_tiler.Detile(m_native_ring->Handle(), ring_offset, size, linear_total, tiles, true);
	for (auto& region: regions) region.bufferOffset += linear.offset;
	image.Upload(regions, linear.buffer, linear.offset, linear.size);
	// Ya en su textura: el refresco del emulador no lo vuelve a subir (y NativeGuardDirty no lo leera de la RAM).
	// (N2+N3d) Sin volver a vigilar sus paginas: el juego solo escribe ahi con el memcpy 0x901459, que intercepta
	// NativeChunkCopy (y si lo deja pasar a la RAM, avisa: NativeUploadHost con ring_offset = kRamWrite). Vigilarlas
	// costaba un 33 % mas de VirtualProtect (log 021220: CPU 8,8 -> 10,7 ms).
	if (image.IsPartiallyCpuDirty() && !image.IsMaybeCpuDirty() && IntersectsRanges(image.CpuDirtyRanges(), address, end))
		image.RefreshRangeComplete(address, end);
	TexNativa::c_direct_bytes.fetch_add(size, std::memory_order_relaxed);
	TexNativa::RingDone(seq, m_scheduler.CurrentTick()); // reutilizable cuando la GPU acabe
	NativeDropRam(address, end);
	return true;
}

void TextureCache::NativeRingPrepare() {
	if (!TexNativa::On() || m_native_ring != nullptr) return;
	m_native_ring = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0, AllFlags, TexNativa::kRingBytes);
	const auto mapped = m_native_ring->Mapped();
	if (mapped.size() < TexNativa::kRingBytes) return; // sin mapa: los trozos van a la RAM como siempre
	std::lock_guard<std::mutex> lock(TexNativa::PMutex());
	TexNativa::TheRing().size = TexNativa::kRingBytes;
	TexNativa::TheRing().base.store(mapped.data(), std::memory_order_release);
}

void TextureCache::NativeRingCollect() {
	TexNativa::RingCollect([&](uint64_t tick) { return m_scheduler.IsFree(tick); });
	// (11-25d, revision 4) Las imagenes borradas/aparcadas desde el grabador: sus bytes "solo en la GPU" a la RAM y retirada.
	if (!g_native_restore_later.load(std::memory_order_acquire) || CommandScheduler::ThreadRecorder() != nullptr ||
	    CommandScheduler::InDeferredOperation())
		return;
	std::scoped_lock lock {m_lock};
	std::vector<std::pair<ImageId, uint64_t>> todo;
	todo.swap(NativeRestoreLater());
	g_native_restore_later.store(false, std::memory_order_release);
	for (const auto& [id, desde]: todo) {
		auto* image = m_slot_images.try_get(id);
		if (image == nullptr || image->registered) continue; // (registrada otra vez: TakeParkedImage no la ve, no deberia)
		const uint64_t base = image->info.data.address, size = image->info.data.size;
		// Descarga de su imagen de Vulkan (viva: aun sin retirar). Sin keep_on_fail: si no se puede, se cuenta perdida.
		if (NativeRestoreBeforeDelete(*image, false, desde)) TexNativa::c_later_ok.fetch_add(1, std::memory_order_relaxed);
		// Las imagenes registradas despues en esa memoria creian esos bytes ya en su textura: sucias, de la RAM ya buena.
		for (const auto other: FindImagesInRegion(base, size, false)) {
			auto* owner = m_slot_images.try_get(other);
			if (owner == nullptr) continue;
			if (TryInvalidatePartial(*owner, base, size, TRACKER_PAGE_SIZE)) continue;
			owner->InvalidateCpuWrite(base, size);
			UntrackImage(other, "tex-nativa-later");
		}
		RetireImage(id);
	}
}

bool TextureCache::NativeRestoreBeforeDelete(Image& image, bool keep_on_fail, uint64_t desde) {
	const uint64_t base = image.info.data.address, end = image.info.data.End();
	std::vector<std::pair<uint64_t, uint64_t>> owned;
	uint64_t                                   epoca = 0; // (WB-EPOCA)
	TexNativa::OwnedIn(base, end, owned, &epoca);
	if (owned.empty()) return true;
	if (desde < epoca) epoca = desde; // (WB-EPOCA) imagen sacada antes del registro: cuenta desde entonces
	uint64_t bytes = 0;
	for (const auto& [b, e]: owned) bytes += e - b;
	const auto lost = [&]() {
		// (N2+N3g) Expulsion del recolector: sin descarga posible la imagen se queda (cero perdida).
		if (keep_on_fail) return false;
		TexNativa::c_lost_bytes.fetch_add(bytes, std::memory_order_relaxed);
		TexNativa::RemoveOwned(base, end);
		return false;
	};
	// Descarga sincrona (recolector bajo presion: como mucho NativeEvictMax por pasada): solo en el hilo de la GPU, fuera de grabaciones.
	if (CommandScheduler::ThreadRecorder() != nullptr || CommandScheduler::InDeferredOperation() || image.depth_id) return lost();
	// (11-25d) Los trozos de su rango que aun esperan en el anillo, a la imagen antes de descargarla (si no, su descarga
	// daria los bytes viejos de esos mips).
	if (image.registered) NativeApplyPending(image);
	auto plan = BuildDownload(image);
	auto& download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	if (!plan.valid || image.info.data.size > download.Size() / 2) return lost();
	auto [mapped, offset] = download.Map(image.info.data.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	if (mapped == nullptr) return lost();
	download.Commit();
	DownloadImageData(image, download, offset, image.info.data.size, std::move(plan));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.Handle();
	barrier.offset              = offset;
	barrier.size                = image.info.data.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost,
	                                               {}, 0, nullptr, 1, &barrier, 0, nullptr);
	m_scheduler.FlushAndWait();
	download.Invalidate(offset, image.info.data.size);
	// Solo los rangos "solo en la GPU" (el resto de la imagen ya estaba en la RAM).
	// (WB-EPOCA) Bajo OwnedMutex (el free 0x802f90 y el unmap esperan): solo lo que nadie libero/reutilizo desde OwnedIn.
	// Lo descartado es ya de quien lo libero o reutilizo: ni se escribe ni se quita del registro (una textura nueva en
	// esa memoria conserva sus rangos "solo en la GPU").
	// (TEX-GUARDA) Y solo lo que sigue protegido contra escritura de la CPU desde que fue "solo en la GPU": una escritura
	// del juego (reutilizacion sin free ni unmap) ya lo habria sacado del registro (epoca nueva). Lo no protegible (sin
	// commit) no se escribe nunca: se pierde (sale del registro), jamas pisa la RAM del juego.
	std::vector<std::pair<uint64_t, uint64_t>> escritos, sin_guarda;
	{
		std::lock_guard<std::mutex> olock(TexNativa::OwnedMutex());
		TexGuarda::EmulatorWrite emulator_write; // (memoria nativa: la escritura va a la direccion del juego)
		for (const auto& [b, e]: owned) {
			if (TexNativa::TocadoDesdeLocked(b, e, epoca)) {
				TexNativa::c_wb_descartado_n.fetch_add(1, std::memory_order_relaxed);
				TexNativa::c_wb_descartado_bytes.fetch_add(e - b, std::memory_order_relaxed);
				continue;
			}
			if (!TexGuarda::Intact(b, e)) {
				TexGuarda::c_wb_unguarded.fetch_add(1, std::memory_order_relaxed);
				TexGuarda::c_wb_unguarded_bytes.fetch_add(e - b, std::memory_order_relaxed);
				sin_guarda.emplace_back(b, e);
				continue;
			}
			LibKernel::Memory::WriteBacking(b, mapped + (b - base), e - b, "tex-nativa");
			escritos.emplace_back(b, e);
		}
	}
	for (const auto& [b, e]: sin_guarda) {
		TexNativa::c_lost_bytes.fetch_add(e - b, std::memory_order_relaxed);
		TexNativa::RemoveOwned(b, e);
	}
	for (const auto& [b, e]: escritos) {
		(void)TexNativa::ForgetReleased(b, e, true); // (N5) bytes de la VRAM escritos: respaldo fisico otra vez
		TexNativa::RemoveOwned(b, e);
	}
	TexNativa::c_restored_bytes.fetch_add(bytes, std::memory_order_relaxed);
	return true;
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

namespace {
// (esperas3, KYTY_ESPERAS: measurement only) A write-window fault's time: the wait for the image lock (TextureCache's
// m_lock, a spin lock every window fault takes exclusive) and the whole call (that wait, the image checks, the memory
// tracker's region lock and VirtualProtect). Every 5 s one "[esperas] ventanas" line. Off: one static bool.
struct WindowFaultStats {
	std::atomic<uint64_t> n {0}, lock_ns {0}, lock_max {0}, total_ns {0}, total_max {0}, slow {0}, slow_lock {0};
	std::atomic<int64_t>  last_ns {0};
};
WindowFaultStats g_window_fault_stats;
bool WindowFaultMeasureOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_ESPERAS");
		return v != nullptr && v[0] != '\0' && v[0] != '0';
	}();
	return on;
}
void AtomicMax(std::atomic<uint64_t>& a, uint64_t v) {
	uint64_t cur = a.load(std::memory_order_relaxed);
	while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
	}
}
struct WindowFaultTimer {
	bool    on;
	int64_t start = 0, locked = 0;
	static int64_t Now() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}
	WindowFaultTimer(): on(WindowFaultMeasureOn()) {
		if (on) start = Now();
	}
	void Locked() {
		if (on) locked = Now();
	}
	~WindowFaultTimer() {
		if (!on) return;
		auto&          s     = g_window_fault_stats;
		const int64_t  end   = Now();
		const uint64_t total = static_cast<uint64_t>(end - start);
		const uint64_t wait  = locked != 0 ? static_cast<uint64_t>(locked - start) : 0;
		s.n.fetch_add(1, std::memory_order_relaxed);
		s.total_ns.fetch_add(total, std::memory_order_relaxed);
		s.lock_ns.fetch_add(wait, std::memory_order_relaxed);
		AtomicMax(s.total_max, total);
		AtomicMax(s.lock_max, wait);
		if (total >= 100000) {
			s.slow.fetch_add(1, std::memory_order_relaxed);
			if (wait * 2 >= total) s.slow_lock.fetch_add(1, std::memory_order_relaxed);
		}
		int64_t last = s.last_ns.load(std::memory_order_relaxed);
		if (last == 0) {
			s.last_ns.compare_exchange_strong(last, end, std::memory_order_relaxed);
			return;
		}
		if (end - last < 5000000000ll || !s.last_ns.compare_exchange_strong(last, end, std::memory_order_relaxed)) return;
		const uint64_t n = s.n.exchange(0), lw = s.lock_ns.exchange(0), lm = s.lock_max.exchange(0), tt = s.total_ns.exchange(0),
		               tm = s.total_max.exchange(0), sl = s.slow.exchange(0), sll = s.slow_lock.exchange(0);
		std::printf("[esperas] ventanas 5 s: %llu fallos de ventana, dentro %.1f ms (el mayor %.2f ms), de eso esperando el "
		            "cerrojo de imagenes %.1f ms (el mayor %.2f ms); lentos (>= 0.1 ms) %llu, %llu de ellos por el cerrojo\n",
		            static_cast<unsigned long long>(n), static_cast<double>(tt) / 1e6, static_cast<double>(tm) / 1e6,
		            static_cast<double>(lw) / 1e6, static_cast<double>(lm) / 1e6, static_cast<unsigned long long>(sl),
		            static_cast<unsigned long long>(sll));
		std::fflush(stdout);
	}
};
} // namespace

bool BufferCache::TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t begin, uint64_t size) {
	if (!GuestRange {begin, size}.Valid()) return false;
	WindowFaultTimer window_fault_timer; // (esperas3, KYTY_ESPERAS)
	// Follow image -> buffer-region -> page lock order. Hold the image lock
	// through invalidation so registration cannot introduce an alias in between.
	std::unique_lock lock(m_texture_cache.m_lock);
	window_fault_timer.Locked();
	if (!m_texture_cache.FindImagesInRegion(begin, size, true).empty()) return false;
	// (Nor a page a pending image download holds: its write fault goes to the texture cache, which lets it go.)
	if (m_texture_cache.HasPendingDownload(begin, size)) return false;
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
	// No image over the range: neither the lock nor the page walk. (The command processor asks for every indirect
	// register table, ~11K a frame at 1-1, in memory the CPU writes; the locked instruction alone was 0.7% of the GPU
	// thread.) A registration racing a reader without the lock makes no image the GPU wrote.
	if (!m_lock.ReadShared([&] { return MayHaveImagesRead(address, size); })) return false;
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

namespace {

// KYTY_SLOW_LOG_MS: a large image a CPU write or a map made dirty whole, or a partial unmap deleted, and why it took no
// partial dirty state (TextureCache::TryInvalidatePartial).
void LogFullDirty(const char* why, const Image& image, uint64_t address, uint64_t size, bool candidate) {
	if (SlowLog::Threshold() <= 0.0 || image.info.data.size < PartialDirtyMinSize || !image.IsTracked() ||
	    kyty_local_partial_image_dirty_mode.load(std::memory_order_relaxed) == 0) {
		return;
	}
	const auto& info = image.info;
	std::printf("[tsc %llu] FULL-DIRTY %s addr=0x%llx size=0x%llx write=0x%llx+0x%llx candidate=%d "
	            "(depth_id=%d backing=%d samples=%u volume=%d depth=%d stencil_plane=%d compression=%u) "
	            "registered=%d tracked=%d whole=%d take=%d maybe=%d cpu=%d partial=%d gpu=%d buffer=%d "
	            "buffer_ranges=%zu stencil=%d\n",
	            static_cast<unsigned long long>(__rdtsc()), why, static_cast<unsigned long long>(info.data.address),
	            static_cast<unsigned long long>(info.data.size), static_cast<unsigned long long>(address),
	            static_cast<unsigned long long>(size), candidate ? 1 : 0, image.depth_id ? 1 : 0,
	            image.backing.image != nullptr ? 1 : 0, info.samples, info.IsVolume() ? 1 : 0, info.IsDepth() ? 1 : 0,
	            info.HasStencil() ? 1 : 0, static_cast<uint32_t>(info.metadata.compression), image.registered ? 1 : 0,
	            image.IsTracked() ? 1 : 0,
	            image.track_addr == info.data.address && image.track_addr_end == info.data.End() ? 1 : 0,
	            image.CanTakePartialDirty() ? 1 : 0, image.IsMaybeCpuDirty() ? 1 : 0,
	            image.IsDefinitelyCpuDirty() ? 1 : 0, image.IsPartiallyCpuDirty() ? 1 : 0,
	            image.IsGpuModified() ? 1 : 0, image.IsBufferModified() ? 1 : 0, image.BufferModifiedRanges().size(),
	            image.IsStencilModified() ? 1 : 0);
	std::fflush(stdout);
}

} // namespace

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size, bool written_pages) {
	const auto page_begin = address & ~(TRACKER_PAGE_SIZE - 1);
	const auto page_end   = (address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (written_pages && owner->IsGpuModified() && owner->Overlaps(address, size)) {
			// (KYTY_NATIVE_MEMORY's whole written pages: an image GPU work wrote, met only on its first or last page, which
			// it shares with other memory. The game does not write into what it has the GPU render; its write was beside
			// the image, as its fault would have shown (a byte outside it). Invalidated, the image would be uploaded
			// again from guest memory, which never had what the GPU rendered: garbage blocks, flicker.)
			const auto image_begin = owner->info.data.address;
			const auto image_end   = owner->info.data.End();
			const auto inner_begin = (image_begin + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
			const auto inner_end   = image_end & ~(TRACKER_PAGE_SIZE - 1);
			const auto hit_begin   = std::max(address, image_begin);
			const auto hit_end     = std::min(address + size, image_end);
			if (!(inner_begin < inner_end && hit_begin < inner_end && inner_begin < hit_end)) continue;
		}
		if (owner->Overlaps(address, size)) {
			{
				const auto base = owner->info.data.address;
				const auto from = std::max(address, base) - base;
				NoteDepthLayersDirtied(*owner, base + from / PartialDirtyGranule * PartialDirtyGranule,
				                       base + ((std::max(address + size, base + 1) - base - 1) / PartialDirtyGranule + 1) *
				                                  PartialDirtyGranule,
				                       LayerDirtyByWrite);
			}
			if (TryInvalidatePartial(*owner, address, size, PartialDirtyGranule)) {
				continue;
			}
			LogFullDirty("write", *owner, address, size, PartialDirtyCandidate(*owner));
			DepthArrayEvent("a CPU write (whole image dirty)", *owner, address, size, nullptr);
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
	found->second.clear_mask        = UINT32_MAX;
	found->second.fill_known        = false;
	found->second.sample_depth_done = 0;
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
	found->second.clear_mask        = UINT32_MAX;
	found->second.fill_value        = fill_value;
	found->second.fill_known        = true;
	found->second.sample_depth_done = 0;
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
	found->second.sample_depth_done &= ~(1u << slice); // (KYTY_SAMPLE_PENDING_HTILE_CLEAR: a new clear, or consumed)
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
	RamNativeTex::ProbeMap(address, address + size); // (ram1b: the first-use probe)
	// Memory mapped under a cached image replaces its contents without a CPU write through the
	// watched pages (the new pages may have been filled through another mapping).
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		// (ram1b, KYTY_RAM_NATIVE_TEX) A sparse array: the layers the map commits get video memory (before the upload the
		// dirty state below brings).
		if (owner->IsSparse() && GuestGpu::IsGpuThread()) owner->SparseUpdate(address, address + size);
		NoteDepthLayersDirtied(*owner, address & ~(TRACKER_PAGE_SIZE - 1), address + size, LayerDirtyByMap);
		if (TryInvalidatePartial(*owner, address, size, TRACKER_PAGE_SIZE)) {
			continue;
		}
		LogFullDirty("map", *owner, address, size, PartialDirtyCandidate(*owner));
		DepthArrayEvent("a map (whole image dirty)", *owner, address, size, nullptr);
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
	TexNativa::RemoveOwned(address, end); // (N2+N3) el juego libero esos mips: ya no hay nada que guardar
	// (KYTY_CUBE_SHADOW_FIX: a shadow pool's parked twin goes with its memory, not 1800 flips later.)
	std::erase_if(m_parked, [&](const auto& item) {
		if (item.second.keep <= 1 || item.first < address || item.first >= end) return false;
		RetireImage(item.second.id);
		return true;
	});
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
	RamNativeTex::ProbeEnd(address, address + size, false); // (ram1b: the first-use probe)
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		// (ram1b, KYTY_RAM_NATIVE_TEX) A sparse array keeping its other layers: the layers decommitted lose their memory.
		if (owner->IsSparse() && GuestGpu::IsGpuThread() &&
		    (address > owner->info.data.address || address + size < owner->info.data.End()))
			owner->SparseUpdate(address, address + size, address, address + size);
		// A texture pool the game streams through keeps its other layers: the game unmaps one
		// layer's memory at a time. The unmapped span becomes dirty (read back as it is when
		// the image is next used) and stays unwatched until then; MapMemory marks it again
		// when memory returns there.
		// (COV4) Imagen nativa dispersa por mip (CTexture): el juego baja de trozo con op 2 (0x7f9f70) sobre los mips
		// sobrantes y 0x901540 rehace el mip base del T#. SparseUpdate (arriba) ya desligo esos mips; la imagen se QUEDA.
		// Antes caia en TryInvalidatePartial (falla: sus paginas no estan vigiladas, bytes solo en la GPU) y se borraba:
		// log 20261126-211247: mips desligados 2713 ~ 2716 trozos 'imagen ya creada (no encontrada o borrada)' (2354 MB a
		// la RAM) al volver a subir de trozo ~34 s despues. Sin marcar sucio: el rango ya no esta comprometido (nada que
		// leer de la RAM) y el siguiente trozo de 0x901330 liga el mip y va directo.
		if (TexNativa::On() && IsNativeSparse(*owner) && GuestGpu::IsGpuThread() &&
		    (address > owner->info.data.address || address + size < owner->info.data.End())) {
			TexNativa::c_unmap_kept.fetch_add(1, std::memory_order_relaxed);
			++slow_partial;
			continue;
		}
		if (address > owner->info.data.address || address + size < owner->info.data.End()) {
			NoteDepthLayersDirtied(*owner, address & ~(TRACKER_PAGE_SIZE - 1), address + size, LayerDirtyByUnmap);
			if (TryInvalidatePartial(*owner, address, size, TRACKER_PAGE_SIZE)) {
				LiveCounters::Add(LiveCounters::PartialUnmaps);
				++slow_partial;
				continue;
			}
			LogFullDirty("unmap", *owner, address, size, PartialDirtyCandidate(*owner));
		}
		DepthArrayEvent("an unmap (image deleted)", *owner, address, size, nullptr);
		++slow_deleted;
		slow_bytes += owner->info.data.size;
		if (owner->info.data.size > slow_largest) {
			slow_largest         = owner->info.data.size;
			slow_largest_address = owner->info.data.address;
		}
		if (owner->IsGpuModified()) {
			owner->ClearGpuModified();
		}
		if (TexNativa::On() && IsNativeSparse(*owner)) TexNativa::c_unmap_del.fetch_add(1, std::memory_order_relaxed); // (COV4)
		DeleteImage(id);
	}
}

void TextureCache::EraseRetired() {
	std::erase_if(m_retired, [&](const auto& retired) {
		if (!Spec::PacketsPassed(retired.second)) return false;
		if (const auto* gone = m_slot_images.try_get(retired.first)) RtNativo::Baja(gone->serial); // (RT-NATIVO)
		m_slot_images.erase(retired.first);
		m_erase_epoch.fetch_add(1, std::memory_order_release);
		return true;
	});
}

// Per flip, whatever the memory use (the collector below runs only near the video memory budget, never on a 32 GB GPU):
// images no draw or dispatch used for RetireTicks flips go, at most MaxRetirements a flip, as stale buffers do
// (BufferCache::RetireStaleBuffers). Kept, their page watchers took a fault at every write the game made when it gave
// that memory to other data (the third loop of the user's 1-1 route: partial dirty faults 17 -> 164 a frame, write
// faults 73 -> 232), and they held their video memory for good. One the game draws again is made again from its bytes.
// Passed over (to the back of the list): images whose bytes only the GPU has (no download here), depth images with a
// stencil association (deleting depth deletes it too), unregistered ones.
// Texture pools of the world the game is in stay where the budget is 12 GiB or more (the collector below takes them,
// oldest first, when it runs): one-level arrays of ChunkedUploadMin and up that shaders only sample. The game samples a
// pool through views from a few first layers to a last one that moves as it streams, and leaves one for minutes as the
// camera moves on: retired, the next view made it again whole from guest memory (2-2 and 5-2 tours: 1.3-2 GB, 150-460
// ms frames, 2-3 times a map), where the image kept serves it or gives its layers on the GPU (HoldLayers). A kept
// pool's watchers take a fault per MiB the game writes at most (partial dirty state). One last used before the game
// entered its world (SceneTags) goes as before: the memory of the worlds left comes back.
// (VRAM-FREE) Imagenes nativas de CTexture destruidas (ver TexNativa::OnEngineFree): fuera, con su VRAM al pool. La
// CTexture ya no existe (el juego no la dibuja mas): nada que guardar, sus rangos "solo en la GPU" salen del registro
// (no se escriben en la RAM: el juego ya puede usar esa memoria para otra cosa). Si una CTexture nueva se registro
// despues en memoria que solapa, no se toca nada (sus bytes son de la nueva).
void TextureCache::DropFreedNativeImages() {
	if (!TexNativa::On() || !TexNativa::g_freed_pending.load(std::memory_order_acquire)) return;
	std::vector<TexNativa::Liberado> q;
	{
		std::lock_guard<std::mutex> lock(TexNativa::FreedMutex());
		q.swap(TexNativa::FreedQueue());
		TexNativa::g_freed_pending.store(false, std::memory_order_release);
	}
	auto reg_seq = [](uint64_t base) {
		std::lock_guard<std::mutex> lock(TexNativa::Mutex());
		const auto it = TexNativa::RegSeq().find(base);
		return it != TexNativa::RegSeq().end() ? it->second : uint64_t {0};
	};
	for (const auto& f: q) {
		const auto images = FindImagesInRegion(f.ptr, 1, false);
		for (const auto id: images) {
			const auto* owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || !IsNativeSparse(*owner)) continue;
			const uint64_t b = owner->info.data.address, e = owner->info.data.End();
			if (reg_seq(b) > f.seq) {
				TexNativa::c_freed_skip.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			bool newer = false;
			for (const auto other: FindImagesInRegion(b, e - b, false)) {
				const auto* o = m_slot_images.try_get(other);
				if (other != id && o != nullptr && o->registered && IsNativeSparse(*o) && reg_seq(o->info.data.address) > f.seq) {
					newer = true;
					break;
				}
			}
			if (newer) {
				TexNativa::c_freed_skip.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			const uint64_t bytes = e - b;
			TexNativa::RemoveOwned(b, e);
			if (owner->IsGpuModified()) m_slot_images[id].ClearGpuModified();
			DeleteImage(id);
			TexNativa::c_freed_imgs.fetch_add(1, std::memory_order_relaxed);
			TexNativa::c_freed_mb.fetch_add(bytes >> 20u, std::memory_order_relaxed);
		}
	}
}

void TextureCache::RetireStaleImages(uint64_t tick) {
	DropFreedNativeImages(); // (VRAM-FREE)
	constexpr uint64_t RetireTicks    = 1200;
	constexpr size_t   MaxRetirements = 16;
	constexpr uint64_t NativeKeepTicks = 18000; // (COV3)
	if (const auto generation = SceneTags::Generation(); generation != m_scene_generation) {
		m_scene_generation = generation;
		m_scene_tick       = tick;
	}
	if (tick <= RetireTicks) return;
	thread_local std::vector<ImageId> stale, kept;
	stale.clear();
	kept.clear();
	const bool keep_pools = !m_graphics.small_video_memory;
	m_lru_cache.ForEachItemBelow(tick - RetireTicks, [&](ImageId id) {
		const auto* owner = m_slot_images.try_get(id);
		if (owner == nullptr || !owner->registered) return false;
		// (N2+N3) Fija: sus bytes "solo en la GPU" viven mientras el juego tenga el mip comprometido (como en la PS5).
		// (N2+N3g) Aqui no hay presion de memoria: solo RunGarbageCollector / ReclaimImageMemory la expulsan.
		if (TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End())) return false;
		// (COV3) Imagen nativa dispersa (creada al crear su CTexture, 0x8fe750): se queda hasta ~5 min sin uso. Log 205533:
		// los trozos de 0x901330 llegan de media 25 s despues (log 200221: 103 s) y este retiro sin presion la borraba a los
		// 1200 ticks (~20 s): 2078 / 9477 MB 'imagen ya creada (no encontrada o borrada)' a la RAM. Sin trozos no liga
		// memoria; con presion de VRAM RunGarbageCollector la sigue expulsando como antes.
		if (TexNativa::On() && IsNativeSparse(*owner) && m_lru_cache.TickOf(owner->lru_id) + NativeKeepTicks > tick) return false;
		// (Left at its age: the collector's.)
		if (keep_pools && owner->info.resources.levels == 1 && owner->info.resources.layers > 1 &&
		    owner->info.data.size >= ChunkedUploadMin && GuestTexture(*owner) &&
		    m_lru_cache.TickOf(owner->lru_id) >= m_scene_tick)
			return false;
		if (owner->depth_id || owner->IsGpuModified())
			kept.push_back(id);
		else
			stale.push_back(id);
		return stale.size() == MaxRetirements;
	});
	for (const auto id: kept) TouchImage(m_slot_images[id]);
	for (const auto id: stale) {
		DeleteImage(id);
		LiveCounters::Add(LiveCounters::GcImageDeletes);
	}
}

[[gnu::noinline]] void TextureCache::LogPcMemory(uint64_t tick) {
	if (tick % 60 != 30) return;
	const auto     use       = LocalPlatform::ProcessMemory();
	const auto     available = LocalPlatform::AvailablePhysicalMemory();
	const uint64_t step      = uint64_t {1} << 30u;
	const auto     moved     = [&](uint64_t now, uint64_t logged) { return now > logged + step || now + step < logged; };
	const bool     short_of_ram = available != 0 && available < (uint64_t {2} << 30u);
	if (!moved(use.working_set, m_logged_working_set) && !moved(use.private_bytes, m_logged_private) &&
	    short_of_ram == m_logged_short_of_ram)
		return;
	m_logged_working_set  = use.working_set;
	m_logged_private      = use.private_bytes;
	m_logged_short_of_ram = short_of_ram;
	std::printf("PC memory (flip %llu): emulator working set %llu MiB, private %llu MiB; the system has %llu MiB of %llu "
	            "MiB left\n",
	            static_cast<unsigned long long>(tick), static_cast<unsigned long long>(use.working_set >> 20u),
	            static_cast<unsigned long long>(use.private_bytes >> 20u), static_cast<unsigned long long>(available >> 20u),
	            static_cast<unsigned long long>(LocalPlatform::PhysicalMemory() >> 20u));
	if (const auto* fresh = FreshMemory::Get())
		std::printf("Fresh memory: %llu MiB of uploads were never-written bytes (zeros, not read); %llu MiB watched, %llu MiB "
		            "written at a first write, %llu MiB by other writes\n",
		            static_cast<unsigned long long>(fresh->zero_upload_bytes.load() >> 20u),
		            static_cast<unsigned long long>(fresh->watched_bytes.load() >> 20u),
		            static_cast<unsigned long long>(fresh->faults.load() >> 20u),
		            static_cast<unsigned long long>(fresh->marked.load() >> 20u));
	std::fflush(stdout);
}

// (ram1b) KYTY_RAM_NATIVE_TEX=1|2 (ram-native-tex.h). GPU thread, once a flip, m_lock held (RunGarbageCollector).
// Every 120 flips (between OffloadScan's): (b) the guest copy of what the cache's images hold - a texture array's
// committed layers (an image of the VirtualMemoryGPU zone with layers, only sampled: Ghidra 01.007.000, its CPU writes
// the layers and never reads them back) and the render targets (GPU-written images of the RTShared/RTPersist arenas:
// PS5CRenderDisplay 0x90ad80 reserves them for GPU work only) -, clean at two scans in a row (no CPU or GPU buffer write
// pending, tracked; a target with no read back pending), mapped, outside the pages it stopped watching and, for an
// array, outside every other GPU-written image: taken as cold ranges of up to 1 MiB (texture-offload.h, RntTakeNow), the
// RAM18 thread writing them to the disk store and dropping them; any touch reads them back from the file. Up to
// KYTY_RAM_NATIVE_TEX_MB (default 512) a scan. =2 takes nothing and counts what it would take.
// Every 5 s: the "[rnt]" line and (=2) one verification (RntVerify).
void TextureCache::RntScan(uint64_t tick) {
	using Clock         = std::chrono::steady_clock;
	const int     mode  = RamNativeTex::Mode();
	static auto   last  = Clock::now();
	constexpr double MiB = 1024.0 * 1024.0;
	if (Clock::now() - last >= std::chrono::seconds(5)) {
		last = Clock::now();
		if (mode == 2) RntVerify();
		auto&      sp = RamNativeTex::Sparse();
		auto&      vf = RamNativeTex::Verify();
		const auto bound = sp.bound_bytes.load(), dense = sp.dense_bytes.load(), chunks = sp.chunk_bytes.load();
		TextureOffload::RntView view;
		const bool read = TextureOffload::RntRead(&view);
		char       probe[640];
		RamNativeTex::ProbeLine(probe, sizeof(probe));
		std::printf("[rnt] 5 s modo=%d | VRAM arrays dispersos: %llu imagenes (hechas %llu, densas por rechazo %llu), ligado %.0f MiB "
		            "(reservado %.0f) de %.0f MiB densos -> ahorro %.0f MiB; capas +%llu -%llu en %llu llamadas (%.1f ms); capas sin "
		            "memoria %llu | RAM %s: soltado ahora capas %.0f + RT %.0f MiB (pico %.0f); tomados capas %.0f / RT %.0f MiB; al "
		            "disco %.0f MiB, en cero %.0f MiB, no escritos %llu; vueltos por un toque: capas %llu (%.0f MiB), RT %llu (%.0f MiB), "
		            "%.1f ms en total (el mayor %.1f ms), edad <10 s %llu, <60 s %llu, mas %llu; PERDIDAS %llu; disco %.0f/%.0f MiB | "
		            "=2 soltaria: capas %.0f + RT %.0f MiB; verificacion: pasadas %llu, capas %llu, iguales %llu, DISCREPANCIA %llu, "
		            "omitidas %llu, formato sin prueba %llu | %s\n",
		            mode, static_cast<unsigned long long>(sp.images.load()), static_cast<unsigned long long>(sp.made.load()),
		            static_cast<unsigned long long>(sp.refused.load()), bound / MiB, chunks / MiB, dense / MiB,
		            dense > chunks ? (dense - chunks) / MiB : 0.0, static_cast<unsigned long long>(sp.layers_bound.load()),
		            static_cast<unsigned long long>(sp.layers_unbound.load()), static_cast<unsigned long long>(sp.bind_calls.load()),
		            sp.bind_us.load() / 1000.0, static_cast<unsigned long long>(sp.alloc_failed.load()),
		            read ? "" : "(sin leer: cerrojo ocupado)", view.now[0] / MiB, view.now[1] / MiB, view.peak / MiB,
		            view.taken_bytes[0] / MiB, view.taken_bytes[1] / MiB, view.written_bytes / MiB, view.zero_bytes / MiB,
		            static_cast<unsigned long long>(view.failed), static_cast<unsigned long long>(view.reads[0]), view.read_bytes[0] / MiB,
		            static_cast<unsigned long long>(view.reads[1]), view.read_bytes[1] / MiB, view.read_us / 1000.0,
		            view.read_us_max / 1000.0, static_cast<unsigned long long>(view.ages[0]), static_cast<unsigned long long>(view.ages[1]),
		            static_cast<unsigned long long>(view.ages[2]), static_cast<unsigned long long>(view.lost), view.file_used / MiB,
		            view.file_size / MiB, RamNativeTex::WouldRelease(0).load() / MiB, RamNativeTex::WouldRelease(1).load() / MiB,
		            static_cast<unsigned long long>(vf.runs.load()), static_cast<unsigned long long>(vf.layers.load()),
		            static_cast<unsigned long long>(vf.equal.load()), static_cast<unsigned long long>(vf.discrepancy.load()),
		            static_cast<unsigned long long>(vf.skipped.load()), static_cast<unsigned long long>(vf.unproven_format.load()), probe);
		// (ram2, KYTY_RAM2) Its line: the last scan, what is out of RAM, the touches that brought it back and their check.
		if (const int ram2 = RamNativeTex::Ram2Mode(); ram2 != 0) {
			auto&                    rs = RamNativeTex::Ram2Scan();
			TextureOffload::Ram2View r2;
			const bool               r2_read = TextureOffload::Ram2Read(&r2);
			const uint64_t           scans   = rs.scans.load();
			std::printf("[ram2] 5 s modo=%d (sin imagen %d s o mas desde el mapeo) | ultima pasada: esperan %.0f MiB, mirados %.0f, "
			            "de ellos de otro asignador (no VirtualMemo) %.0f, en zona vetada %.0f, bajo imagen %.0f, ya tomados %.0f, "
			            "retenidos por un toque reciente %.0f, fuera de RAM o protegidos %.0f, %s %.0f MiB; pasada %.2f ms (media "
			            "%.2f, la mayor %.2f) | RAM%s: soltado ahora %.0f MiB (pico %.0f); "
			            "tomados %llu (%.0f MiB); al disco %.0f MiB, en cero %.0f MiB, no escritos "
			            "%llu; desmapeados fuera %.0f MiB | vueltos por un toque %llu (%.0f MiB), %.1f ms en total (el mayor %.2f ms), "
			            "edad desde la toma <10 s %llu, <60 s %llu, mas %llu | verificacion: comprobados %llu (%.0f MiB, %.1f ms), "
			            "iguales %llu, DISCREPANCIA %llu, sin comprobar %llu | PERDIDAS %llu | disco %.0f/%.0f MiB | proteccion: zonas "
			            "vetadas %llu (escrituras del emulador %llu, fallos repetidos %llu, lentos %llu; el fallo mas largo %.1f ms)\n",
			            ram2, RamNativeTex::Ram2AgeSeconds(), rs.waiting.load() / MiB, rs.aged.load() / MiB,
			            rs.other_label.load() / MiB, rs.blocked.load() / MiB, rs.under_image.load() / MiB, rs.taken.load() / MiB, rs.held.load() / MiB, rs.not_ram.load() / MiB, ram2 == 1 ? "propuestos" : "=2 soltaria", rs.proposed.load() / MiB,
			            rs.scan_us_last.load() / 1000.0, scans != 0 ? rs.scan_us.load() / 1000.0 / static_cast<double>(scans) : 0.0,
			            rs.scan_us_max.load() / 1000.0, r2_read ? "" : " (sin leer: cerrojo ocupado)", r2.now / MiB, r2.peak / MiB,
			            static_cast<unsigned long long>(r2.taken), r2.taken_bytes / MiB, r2.written_bytes / MiB, r2.zero_bytes / MiB,
			            static_cast<unsigned long long>(r2.failed), r2.unmapped / MiB, static_cast<unsigned long long>(r2.reads),
			            r2.read_bytes / MiB, r2.read_us / 1000.0, r2.read_us_max / 1000.0, static_cast<unsigned long long>(r2.ages[0]),
			            static_cast<unsigned long long>(r2.ages[1]), static_cast<unsigned long long>(r2.ages[2]),
			            static_cast<unsigned long long>(r2.checked), r2.checked_bytes / MiB, r2.check_us / 1000.0,
			            static_cast<unsigned long long>(r2.equal), static_cast<unsigned long long>(r2.discrepancy),
			            static_cast<unsigned long long>(r2.unchecked), static_cast<unsigned long long>(r2.lost), r2.file_used / MiB,
			            r2.file_size / MiB, static_cast<unsigned long long>(RamNativeTex::Ram2Guards().count.load()),
			            static_cast<unsigned long long>(RamNativeTex::Ram2Guards().host_writes.load()),
			            static_cast<unsigned long long>(RamNativeTex::Ram2Guards().repeats.load()),
			            static_cast<unsigned long long>(RamNativeTex::Ram2Guards().slow.load()),
			            RamNativeTex::Ram2Guards().slow_us_max.load() / 1000.0);
		}
		std::fflush(stdout);
	}
	if (tick % 120 != 60 || !TextureOffload::Native()) return;
	if (mode == 1 && !TextureOffload::Ram18On()) return;
	// (ram2, KYTY_RAM2=1|2) The zone's memory no image has used for KYTY_RAM2_AGE_S since its map (the first-use probe's
	// waits, oldest first), mapped, outside every image of the cache (an image over it is RAM16's / (b)'s), not taken,
	// committed plain read-write or read-only with a page in RAM: taken as cold ranges of up to 1 MiB (RntRam2), the RAM18
	// thread writing them to the disk store and dropping them; any touch reads them back and verifies them. Up to
	// KYTY_RAM2_MB (128) a scan, 4 GiB of waits and 768 chunks' page checks at most (render thread: its time is in the
	// line). =2 takes nothing and counts what it would take.
	if (const int ram2 = RamNativeTex::Ram2Mode(); ram2 != 0) {
		using Clock2 = std::chrono::steady_clock;
		const auto r2_start = Clock2::now();
		constexpr uint64_t R2Page = TRACKER_PAGE_SIZE, R2Chunk = uint64_t {1} << 20u, R2Look = uint64_t {4} << 30u;
		const uint64_t r2_budget = ram2 == 2 ? UINT64_MAX : RamNativeTex::Ram2Budget();
		std::vector<std::pair<uint64_t, uint64_t>>           waits, mapped, pieces, scratch;
		std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> r2_chunks;
		RamNativeTex::ProbeAged(static_cast<double>(RamNativeTex::Ram2AgeSeconds()), &waits);
		uint64_t aged = 0, under_image = 0, taken = 0, held = 0, proposed = 0, not_ram = 0, waiting = 0, other_label = 0, blocked = 0;
		// (ram2b) Only the VirtualMemo allocator's memory (the streamed textures): the game's heaps (cpuheap, cpubkt, ...)
		// map 2 MiB blocks in the same address zone, and the CPU and the emulator (file reads) write them - ram2's hang of
		// 11-18 21:26 was a game heap write faulting for ever on a range a host write had marked missed. The ranges of the
		// game's allocators from the memory map's last snapshot (lock-free); none (no snapshot): nothing taken this scan.
		static std::vector<uint64_t> r2_pairs(uint64_t {2} * 16384u);
		std::vector<std::pair<uint64_t, uint64_t>> r2_vm, r2_other;
		{
			static const char* const vm_label[] = {"VirtualMemo"};
			static const char* const other_labels[] = {"cpuheap", "gpuheap", "cpubkt", "cpupage", "gpubkt", "gpupage",
			                                           "DefaultTagg", "RTSharedAll", "RTPersistAl", "ring", "pool", "ctx"};
			const size_t n = Loader::DemonsSoulsMemMap::RangesOf(vm_label, 1, r2_pairs.data(), r2_pairs.size() / 2);
			if (n != SIZE_MAX)
				for (size_t i = 0; i < n; i++)
					if (r2_pairs[2 * i + 1] > r2_pairs[2 * i]) r2_vm.emplace_back(r2_pairs[2 * i], r2_pairs[2 * i + 1]);
			const size_t m = n == SIZE_MAX ? SIZE_MAX
			                               : Loader::DemonsSoulsMemMap::RangesOf(other_labels, sizeof(other_labels) / sizeof(other_labels[0]), r2_pairs.data(),
			                                                                     r2_pairs.size() / 2);
			if (m == SIZE_MAX) r2_vm.clear(); // (no snapshot of both: nothing this scan)
			else
				for (size_t i = 0; i < m; i++)
					if (r2_pairs[2 * i + 1] > r2_pairs[2 * i]) r2_other.emplace_back(r2_pairs[2 * i], r2_pairs[2 * i + 1]);
			for (auto* list: {&r2_vm, &r2_other}) { // (sorted, overlaps merged)
				std::sort(list->begin(), list->end());
				size_t w = 0;
				for (size_t r = 0; r < list->size(); r++) {
					if (w != 0 && (*list)[r].first <= (*list)[w - 1].second) (*list)[w - 1].second = std::max((*list)[w - 1].second, (*list)[r].second);
					else (*list)[w++] = (*list)[r];
				}
				list->resize(w);
			}
		}
		// [lo, hi) pieces kept inside the VirtualMemo ranges and outside every other allocator's (into `out`).
		const auto keep_vm = [&](uint64_t lo, uint64_t hi, std::vector<std::pair<uint64_t, uint64_t>>& out) {
			auto it = std::upper_bound(r2_vm.begin(), r2_vm.end(), std::make_pair(lo, UINT64_MAX));
			if (it != r2_vm.begin()) --it;
			for (; it != r2_vm.end() && it->first < hi; ++it) {
				uint64_t a = std::max(lo, it->first), b = std::min(hi, it->second);
				if (a >= b) continue;
				auto ot = std::upper_bound(r2_other.begin(), r2_other.end(), std::make_pair(a, UINT64_MAX));
				if (ot != r2_other.begin()) --ot;
				for (; ot != r2_other.end() && ot->first < b && a < b; ++ot) {
					if (ot->second <= a) continue;
					if (ot->first > a) out.emplace_back(a, ot->first);
					a = std::max(a, ot->second);
				}
				if (a < b) out.emplace_back(a, b);
			}
		};
		int      checks = ram2 == 2 ? 4096 : 768;
		// A wait that gave nothing to take (all taken, under an image, out of RAM or protected) is looked at again in 30 s,
		// not at every scan (the render thread's time).
		static std::unordered_map<uint64_t, Clock2::time_point> r2_next;
		std::unordered_map<uint64_t, Clock2::time_point>        r2_keep;
		for (const auto& [wait_begin, wait_end]: waits) {
			waiting += wait_end - wait_begin;
			const auto next = r2_next.find(wait_begin);
			if (next != r2_next.end() && r2_start < next->second) {
				r2_keep.emplace(next->first, next->second);
				continue;
			}
			if (aged >= R2Look || proposed >= r2_budget || checks <= 0) continue;
			const uint64_t proposed_before = proposed;
			bool           cut_short       = false;
			Libs::LibKernel::Memory::MappedParts(wait_begin, wait_end - wait_begin, &mapped);
			pieces.clear();
			scratch.clear();
			for (const auto& [at, bytes]: mapped) {
				const uint64_t lo = (std::max(at, wait_begin) + R2Page - 1) & ~(R2Page - 1);
				const uint64_t hi = std::min(at + bytes, wait_end) & ~(R2Page - 1);
				if (lo < hi) scratch.emplace_back(lo, hi);
			}
			uint64_t mapped_bytes = 0, kept_bytes = 0;
			for (const auto& [lo, hi]: scratch) {
				mapped_bytes += hi - lo;
				keep_vm(lo, hi, pieces);
			}
			for (auto& [lo, hi]: pieces) {
				lo = (lo + R2Page - 1) & ~(R2Page - 1);
				hi = hi & ~(R2Page - 1);
				if (hi < lo) hi = lo;
				kept_bytes += hi - lo;
			}
			aged += mapped_bytes;
			other_label += mapped_bytes - std::min(mapped_bytes, kept_bytes);
			// Any image of the cache over it (made before the map, or merged): its memory is the images' paths', not this.
			for (const auto id: FindImagesInRegion(wait_begin, wait_end - wait_begin, false)) {
				const auto* image = m_slot_images.try_get(id);
				if (image == nullptr || image->info.data.Empty()) continue;
				const uint64_t cut_lo = image->info.data.address & ~(R2Page - 1);
				const uint64_t cut_hi = (image->info.data.End() + R2Page - 1) & ~(R2Page - 1);
				scratch.clear();
				for (const auto& [lo, hi]: pieces) {
					if (cut_hi <= lo || cut_lo >= hi) {
						scratch.emplace_back(lo, hi);
						continue;
					}
					under_image += std::min(hi, cut_hi) - std::max(lo, cut_lo);
					if (lo < cut_lo) scratch.emplace_back(lo, cut_lo);
					if (cut_hi < hi) scratch.emplace_back(cut_hi, hi);
				}
				pieces.swap(scratch);
			}
			for (const auto& [lo, hi]: pieces) {
				for (uint64_t at = lo; at < hi && !cut_short;) {
					if (proposed >= r2_budget || checks <= 0) {
						cut_short = true;
						break;
					}
					const uint64_t stop = std::min(hi, (at | (R2Chunk - 1)) + 1);
					const int status = RamNativeTex::Ram2Blocked(at, stop) ? 3 : TextureOffload::Ram2Status(at, stop - at);
					if (status == 3) {
						blocked += stop - at;
					} else if (status == 1) {
						taken += stop - at;
					} else if (status == 2) {
						held += stop - at;
					} else {
						checks--;
						if (!RamNativeTex::Ram2Plain(at, stop - at)) {
							not_ram += stop - at;
						} else {
							r2_chunks.emplace_back(at, stop - at, TextureOffload::RntRam2);
							proposed += stop - at;
						}
					}
					at = stop;
				}
			}
			if (!cut_short && proposed == proposed_before) r2_keep[wait_begin] = r2_start + std::chrono::seconds(30);
		}
		r2_next.swap(r2_keep);
		if (ram2 == 1 && !r2_chunks.empty()) {
			// Their read watchers on before the RAM18 thread writes and drops them (pending upload jobs over them first).
			for (const auto& [begin, size]: TextureOffload::Ram2TakeNow(r2_chunks)) {
				AsyncUpload::DrainRange(begin, size);
				m_page_manager.UpdateReadWatchers<true>(begin, size);
			}
		}
		auto&          rs = RamNativeTex::Ram2Scan();
		const uint64_t us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock2::now() - r2_start).count());
		rs.waiting.store(waiting, std::memory_order_relaxed);
		rs.aged.store(aged, std::memory_order_relaxed);
		rs.under_image.store(under_image, std::memory_order_relaxed);
		rs.taken.store(taken, std::memory_order_relaxed);
		rs.held.store(held, std::memory_order_relaxed);
		rs.other_label.store(other_label, std::memory_order_relaxed);
		rs.blocked.store(blocked, std::memory_order_relaxed);
		rs.not_ram.store(not_ram, std::memory_order_relaxed);
		rs.proposed.store(proposed, std::memory_order_relaxed);
		rs.scan_us.fetch_add(us, std::memory_order_relaxed);
		rs.scan_us_last.store(us, std::memory_order_relaxed);
		rs.scans.fetch_add(1, std::memory_order_relaxed);
		if (us > rs.scan_us_max.load(std::memory_order_relaxed)) rs.scan_us_max.store(us, std::memory_order_relaxed);
	}
	static const uint64_t take_budget = [] {
		const char* value = std::getenv("KYTY_RAM_NATIVE_TEX_MB");
		return (value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : uint64_t {512}) << 20u;
	}();
	const uint64_t budget = mode == 2 ? UINT64_MAX : take_budget; // (=2 counts all it would take)
	constexpr uint64_t Page = TRACKER_PAGE_SIZE, Chunk = uint64_t {1} << 20u;
	struct Candidate {
		uint64_t                                   begin = 0, end = 0;
		uint8_t                                    kind  = 0; // 1 array, 2 render target
		std::vector<std::pair<uint64_t, uint64_t>> holes;
	};
	std::vector<Candidate>                     candidates;
	std::vector<std::pair<uint64_t, uint64_t>> gpu_ranges; // (other GPU-written images: arrays' chunks under them stay)
	std::unordered_map<uint64_t, uint64_t>     seen;
	m_slot_images.ForEach([&](ImageId id, Image& image) {
		if (!image.registered || image.info.data.Empty()) return;
		const uint64_t begin = (image.info.data.address + Page - 1) & ~(Page - 1), end = image.info.data.End() & ~(Page - 1);
		if (end <= begin) return;
		const bool gpu   = GpuWrittenImage(image);
		const int  arena = gpu ? Loader::DemonsSoulsMemMap::RtArenaOf(image.info.data.address, image.info.data.End()) : 0;
		uint8_t    kind  = 0;
		if (gpu && (arena == 1 || arena == 2) && !image.usage.video_out) kind = 2;
		else if (gpu) {
			gpu_ranges.emplace_back(begin, end);
			return;
		} else if (image.usage.texture && !image.usage.video_out && !image.depth_id && image.info.resources.layers >= 2 &&
		           RamNativeTex::InZone(begin, end))
			kind = 1;
		if (kind == 0 || image.IsCpuDirty() || image.IsBufferModified() || !image.IsTracked()) return;
		if (kind == 2 && (m_download_images.contains(id) || HasPendingDownload(image.info.data.address, image.info.data.size) ||
		                  m_buffer_cache.HasGpuDirtyBytes(image.info.data.address, image.info.data.size)))
			return;
		const auto first   = m_rnt_clean.find(image.serial);
		seen[image.serial] = first != m_rnt_clean.end() ? first->second : tick;
		if (first == m_rnt_clean.end()) return;
		candidates.push_back({begin, end, kind, image.untracked_holes});
	});
	m_rnt_clean.swap(seen);
	std::sort(gpu_ranges.begin(), gpu_ranges.end());
	std::vector<std::tuple<uint64_t, uint64_t, uint8_t>> chunks;
	uint64_t                                             proposed = 0, would[2] {};
	std::vector<std::pair<uint64_t, uint64_t>>           parts, pieces, next;
	const auto subtract = [](std::vector<std::pair<uint64_t, uint64_t>>& from, const std::vector<std::pair<uint64_t, uint64_t>>& cut,
	                         std::vector<std::pair<uint64_t, uint64_t>>& scratch) {
		for (const auto& [cut_begin, cut_end]: cut) {
			scratch.clear();
			for (const auto& [lo, hi]: from) {
				if (cut_end <= lo || cut_begin >= hi) {
					scratch.emplace_back(lo, hi);
					continue;
				}
				if (lo < cut_begin) scratch.emplace_back(lo, cut_begin);
				if (cut_end < hi) scratch.emplace_back(cut_end, hi);
			}
			from.swap(scratch);
		}
	};
	for (const auto& candidate: candidates) {
		if (proposed >= budget) break;
		Libs::LibKernel::Memory::MappedParts(candidate.begin, candidate.end - candidate.begin, &parts);
		pieces.clear();
		for (const auto& [at, bytes]: parts) {
			const uint64_t lo = std::max(at, candidate.begin), hi = std::min(at + bytes, candidate.end);
			if (lo < hi) pieces.emplace_back(lo, hi);
		}
		subtract(pieces, candidate.holes, next);
		if (candidate.kind == 1) {
			std::vector<std::pair<uint64_t, uint64_t>> under;
			for (const auto& range: gpu_ranges)
				if (range.first < candidate.end && range.second > candidate.begin) under.push_back(range);
			subtract(pieces, under, next);
		}
		for (const auto& [lo_raw, hi_raw]: pieces) {
			const uint64_t lo = (lo_raw + Page - 1) & ~(Page - 1), hi = hi_raw & ~(Page - 1);
			for (uint64_t at = lo; at < hi && proposed < budget;) {
				const uint64_t stop = std::min(hi, (at | (Chunk - 1)) + 1);
				if (TextureOffload::AnyTakenOver(at, stop - at)) {
					at = stop;
					continue;
				}
				chunks.emplace_back(at, stop - at, candidate.kind);
				proposed += stop - at;
				at = stop;
			}
		}
	}
	// (Images over the same memory - a texture held from a level and the whole one - propose it once.)
	std::sort(chunks.begin(), chunks.end());
	chunks.erase(std::unique(chunks.begin(), chunks.end(),
	                         [](const auto& a, const auto& b) { return std::get<0>(a) == std::get<0>(b); }),
	             chunks.end());
	for (const auto& [begin, size, kind]: chunks) would[kind - 1] += size;
	if (mode == 2) {
		RamNativeTex::WouldRelease(0).store(would[0], std::memory_order_relaxed);
		RamNativeTex::WouldRelease(1).store(would[1], std::memory_order_relaxed);
		return;
	}
	if (chunks.empty()) return;
	// Their read watchers on before the RAM18 thread writes and drops them (pending upload jobs over them first).
	for (const auto& [begin, size]: TextureOffload::RntTakeNow(chunks)) {
		AsyncUpload::DrainRange(begin, size);
		m_page_manager.UpdateReadWatchers<true>(begin, size);
	}
}

// (ram1b, KYTY_RAM_NATIVE_TEX=2) One sparse image (the next by serial, clean, only sampled) checked against the current
// path: a dense image of its description uploaded from guest memory (UploadImage, as InitializeImage's whole upload),
// both downloaded for up to 32 MiB of its bound layers (a rotating window), compared layer by layer by the offload worker
// ("DISCREPANCIA" in the "[rnt]" line, the first ones logged). One at a time. Skipped (counted) when an offload holds part
// of its guest bytes out of RAM (its reference would read them back) or a buffer cannot be had.
void TextureCache::RntVerify() {
	auto& counters = RamNativeTex::Verify();
	std::erase_if(m_rnt_checks, [](const std::unique_ptr<RntCheck>& check) { return check->done.load(std::memory_order_acquire); });
	if (!m_rnt_checks.empty() || !g_offload_alive.load(std::memory_order_acquire) || CommandScheduler::ThreadRecorder() != nullptr)
		return;
	ImageId  chosen {}, first {};
	uint64_t chosen_serial = UINT64_MAX, first_serial = UINT64_MAX;
	m_slot_images.ForEach([&](ImageId id, Image& image) {
		if (!image.IsSparse() || !image.registered || image.IsCpuDirty() || image.IsBufferModified() || image.IsGpuModified() ||
		    image.usage.render_target || image.usage.storage || image.usage.depth_target)
			return;
		if (image.serial > m_rnt_verify_after && image.serial < chosen_serial) chosen = id, chosen_serial = image.serial;
		if (image.serial < first_serial) first = id, first_serial = image.serial;
	});
	if (chosen_serial == UINT64_MAX) chosen = first, chosen_serial = first_serial;
	if (chosen_serial == UINT64_MAX) return;
	m_rnt_verify_after = chosen_serial;
	const ImageInfo info = m_slot_images[chosen].info;
	if (TextureOffload::Overlaps(info.data.address, info.data.size) || TextureOffload::AnyTakenOver(info.data.address, info.data.size)) {
		counters.skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	std::vector<uint32_t> bound;
	SparseBoundLayers(m_slot_images[chosen].backing, &bound);
	if (bound.empty()) return;
	// Every level of a layer, tightly packed in the format's blocks.
	const auto     format      = m_slot_images[chosen].backing.format;
	const uint64_t block_bytes = vk::blockSize(static_cast<VULKAN_HPP_NAMESPACE::Format>(format));
	const auto     block       = vk::blockExtent(static_cast<VULKAN_HPP_NAMESPACE::Format>(format));
	const uint32_t levels      = m_slot_images[chosen].backing.mip_levels;
	const auto     extent      = m_slot_images[chosen].backing.extent;
	if (block_bytes == 0 || block[0] == 0 || block[1] == 0) {
		counters.skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	uint64_t layer_bytes = 0;
	for (uint32_t level = 0; level < levels; level++) {
		const uint64_t w = std::max(extent.width >> level, 1u), h = std::max(extent.height >> level, 1u);
		layer_bytes += ((w + block[0] - 1) / block[0]) * ((h + block[1] - 1) / block[1]) * block_bytes;
	}
	constexpr uint64_t Window = uint64_t {32} << 20u;
	const size_t       count  = std::max<size_t>(1, std::min<size_t>(bound.size(), static_cast<size_t>(Window / std::max<uint64_t>(layer_bytes, 1))));
	static uint64_t    rotate = 0;
	const size_t       start  = static_cast<size_t>(rotate++ % bound.size());
	auto               check  = std::make_unique<RntCheck>();
	std::vector<vk::BufferImageCopy> copies;
	uint64_t                         offset = 0;
	for (size_t i = 0; i < count; i++) {
		const uint32_t layer = bound[(start + i) % bound.size()];
		check->layers.emplace_back(layer, offset, layer_bytes);
		for (uint32_t level = 0; level < levels; level++) {
			const uint32_t      w = std::max(extent.width >> level, 1u), h = std::max(extent.height >> level, 1u);
			vk::BufferImageCopy copy {};
			copy.bufferOffset     = offset;
			copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, level, layer, 1};
			copy.imageExtent      = vk::Extent3D {w, h, 1};
			copies.push_back(copy);
			offset += ((w + block[0] - 1) / block[0]) * ((h + block[1] - 1) / block[1]) * block_bytes;
		}
	}
	check->half = (offset + 255) & ~uint64_t {255};
	// The reference: the same description, dense, uploaded from guest memory now.
	ImageId reference_id {};
	{
		ImageForceDense dense;
		reference_id = m_slot_images.insert(m_graphics, m_scheduler, info, true);
	}
	auto& reference = m_slot_images[reference_id];
	if (reference.CreateFailed() || reference.IsSparse()) {
		RetireImage(reference_id);
		counters.skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	reference.usage   = m_slot_images[chosen].usage;
	reference.binding = m_slot_images[chosen].binding;
	bool refillable   = false;
	const auto [source, source_offset] = m_buffer_cache.ObtainBufferForImage(info.data.address, info.data.size, &refillable);
	if (source == nullptr) {
		RetireImage(reference_id);
		counters.skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	UploadImage(reference, *source, source_offset);
	check->buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, check->half * 2);
	m_slot_images[reference_id].Download(copies, check->buffer->Handle(), 0, check->half);
	for (auto& copy: copies) copy.bufferOffset += check->half;
	m_slot_images[chosen].Download(copies, check->buffer->Handle(), check->half, check->half);
	RetireImage(reference_id);
	check->proven = m_offload_good_formats.contains(OffloadFormat(m_slot_images[chosen]));
	char what[160];
	std::snprintf(what, sizeof(what), "0x%llx %ux%u capas=%u niveles=%u formato=%d", static_cast<unsigned long long>(info.data.address),
	              extent.width, extent.height, m_slot_images[chosen].backing.layers, levels, static_cast<int>(format));
	check->what = what;
	counters.runs.fetch_add(1, std::memory_order_relaxed);
	auto* raw = check.get();
	m_rnt_checks.push_back(std::move(check));
	m_scheduler.DeferPriorityOperation([raw] { OffloadWorker::Post([raw] {
	RtNativo::Informe(); // (RT-NATIVO)
		auto& verify = RamNativeTex::Verify();
		if (!g_offload_alive.load(std::memory_order_acquire)) {
			raw->done.store(true, std::memory_order_release);
			return;
		}
		raw->buffer->Invalidate(0, raw->buffer->Size());
		const auto* mapped = raw->buffer->Mapped().data();
		static std::atomic<uint32_t> reports {0};
		for (const auto& [layer, at, bytes]: raw->layers) {
			const bool equal = XXH3_64bits(mapped + at, bytes) == XXH3_64bits(mapped + raw->half + at, bytes);
			verify.layers.fetch_add(1, std::memory_order_relaxed);
			(equal ? verify.equal : verify.discrepancy).fetch_add(1, std::memory_order_relaxed);
			if (!equal && reports.fetch_add(1, std::memory_order_relaxed) < 16)
				std::printf("[rnt] DISCREPANCIA: capa %u de %s distinta de la via actual (formato %s)\n", layer, raw->what.c_str(),
				            raw->proven ? "probado" : "sin prueba de descarga");
		}
		if (!raw->proven) verify.unproven_format.fetch_add(1, std::memory_order_relaxed);
		std::fflush(stdout);
		raw->done.store(true, std::memory_order_release);
	}); });
}

void TextureCache::RunGarbageCollector() {
	EraseRetired();
	std::scoped_lock lock {m_lock};
	const uint64_t   tick = m_gc_tick++;
	// Parked images not taken back since the flip before: the game left that use of the memory.
	std::erase_if(m_parked, [&](auto& item) {
		// (W10) La gemela de un pool de sombras (keep == CubeTwinKeep) ya no caduca por flips: solo se va con su memoria
		// (FreeMemory) o con presion real (ReclaimImageMemory). Antes, a los 1800 flips se soltaba y el siguiente cambio
		// profundidad<->color creaba una imagen nueva (replacement=new).
		if (item.second.keep == CubeTwinKeep) {
			if (tick > item.second.tick + CubeTwinKeep) {
				static std::atomic<uint32_t> w10_told {0};
				if (w10_told.fetch_add(1, std::memory_order_relaxed) < 4) {
					std::printf("[w10] gemela de sombras conservada pasados %llu flips (antes se soltaba): addr=0x%llx\n",
					            static_cast<unsigned long long>(tick - item.second.tick),
					            static_cast<unsigned long long>(item.first));
					std::fflush(stdout);
				}
				item.second.tick = tick; // (sin repetir la linea en cada flip)
			}
			return false;
		}
		if (item.second.tick + item.second.keep >= tick) return false;
		RetireImage(item.second.id);
		return true;
	});
	GestorPass(tick); // (M7) candidatas del gestor central, copias RAM, linea [tex-dedup]
	// (GPUNAT-RT) RT de las arenas solo en VRAM: su copia en la RAM del juego fuera (ver namespace GpuNatRt).
	if (GpuNatRt::On()) {
		GpuNatRt::Frame();
		if (tick % 60 == 30 && LibKernel::Memory::NativeMemoryEnabled()) {
			const auto t0 = std::chrono::steady_clock::now();
			struct R {
				uint64_t b, e, serial;
				bool     ok, nueva;
			};
			std::vector<R> all;
			m_slot_images.ForEach([&](ImageId /*id*/, const Image& image) {
				if (!image.registered || image.info.data.Empty()) return;
				const bool ok = !image.IsSparse() && image.backing.image != nullptr && RtNativo::Es(image) && image.SafeToDownload();
				all.push_back({image.info.data.address, image.info.data.End(), image.serial, ok,
				               ok && !GpuNatRt::Hechos().contains(image.serial)});
			});
			constexpr uint64_t Tope = uint64_t {128} << 20u;
			uint64_t           hecho = 0;
			for (const auto& c: all) {
				if (!c.nueva || hecho >= Tope) continue;
				const uint64_t b = (c.b + 0xfffu) & ~uint64_t {0xfff}, e = c.e & ~uint64_t {0xfff};
				bool           seguro = e > b;
				for (const auto& o: all) // todas las imagenes sobre esas paginas: RT nativos con la GPU como duena
					if (seguro && o.b < e && o.e > b && !o.ok) seguro = false;
				seguro = seguro && !HasPendingDownload(b, e - b) && !TextureOffload::Overlaps(b, e - b) &&
				         !(TexNativa::On() && TexNativa::AnyOwned(b, e));
				if (seguro && m_buffer_cache.IsRegionRegistered(b, e - b))
					seguro = !m_buffer_cache.HasGpuDirtyBytes(b, e - b) && !m_buffer_cache.IsRegionCpuModified(b, e - b);
				GpuNatRt::Hechos().insert(c.serial); // (una vez por imagen, se suelte o no)
				if (!seguro) {
					GpuNatRt::c_saltadas++;
					continue;
				}
				uint64_t       refused = 0;
				const uint64_t done    = LibKernel::Memory::NativeOffloadDrop(b, e - b, &refused);
				GpuNatRt::c_imgs++;
				GpuNatRt::c_soltado += done;
				GpuNatRt::c_rechazado += refused;
				GpuNatRt::c_sin_commit += (e - b) - std::min(e - b, done + refused);
				hecho += e - b;
			}
			const auto us = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
			GpuNatRt::c_us += us;
			GpuNatRt::c_us_max = std::max(GpuNatRt::c_us_max, us);
		}
	}
	RetireStaleImages(tick);
	// (KYTY_BLOCK_CLASS_OVERLAP_REPLACE: past its first 32 lines, the replacements since the last, every 60 flips.)
	if (m_block_class_replaced != m_block_class_logged && tick >= m_block_class_tick + 60) {
		std::printf("Texture cache: %llu images of the other block class replaced since the last line (%llu in all)\n",
		            static_cast<unsigned long long>(m_block_class_replaced - m_block_class_logged),
		            static_cast<unsigned long long>(m_block_class_replaced));
		std::fflush(stdout);
		m_block_class_logged = m_block_class_replaced;
		m_block_class_tick   = tick;
	}
	ShadowPoolTrace::Flip(tick);
	// (11-21) The texture offload pages clean textures out to disk: only on PCs of 20 GB or less (the low-RAM profile) and
	// only while the PC is almost out of RAM (under 1.5 GiB available, until 3 GiB again). Before, it ran every frame
	// whatever the free RAM, and its page-outs and restores were the regular hitches on 16 GB.
	static bool s_offload_pressure = false;
	{
		const uint64_t avail = RamGuard::LowRam() ? RamGuard::AvailablePhysical() : 0;
		if (avail == 0) s_offload_pressure = false;
		else if (avail < (uint64_t {1536} << 20u)) s_offload_pressure = true;
		else if (avail > (uint64_t {3072} << 20u)) s_offload_pressure = false;
	}
	// (M7) El gestor (RAM critica, expulsor de copias RAM) o ram-guard (critico) piden la pasada ya, sin esperar al tick.
	GestorTex::forzar_scan = GestorTex::ram_prisa.exchange(false, std::memory_order_relaxed);
	if (TextureOffload::Enabled() && (s_offload_pressure || GestorTex::forzar_scan)) OffloadScan(tick);
	GestorTex::forzar_scan = false;
	// (picos1) The same for the disk store of texture arrays and render targets (RAM17/RAM18): with it taking them every
	// scan, a touch of the 320 MiB shadow/texture pool waited for them back - 30-100 ms frames at any RAM, none with it off.
	// (picos6) Not even under pressure unless KYTY_RNT_SCAN=1: on a 16 GB PC near its limit it took layers of the texture
	// pools (80-256 layer arrays) and their touches read them back - 130 ms to 2.9 s frames. 8 GB kept in RAM instead.
	static const bool rnt_scan = [] { const char* v = std::getenv("KYTY_RNT_SCAN"); return v != nullptr && *v == '1'; }();
	if (RamNativeTex::On() && s_offload_pressure && rnt_scan) RntScan(tick); // (ram1b)
	// (BUF-NATIVO) KYTY_BUF_NATIVO=1 (por defecto; local/buf-nativo.h): las mallas registradas por los ganchos de 0xaa06a0
	// hace mas de KYTY_BUF_NATIVO_DELAY_S (la cache de buferes ya las subio a la VRAM) se toman como rangos frios: el hilo
	// RAM18 las copia al almacen (VRAM) y suelta su RAM; un toque las lee de vuelta. Sin depender de la presion de RAM (sus
	// datos no los vuelve a leer la CPU: lo mide la linea). Cada 30 flips como mucho una pasada; la linea cada 30 s.
	if (BufNativo::On() && BufNativo::Get().hooked.load(std::memory_order_acquire)) {
		static bool s_bn_ready = false;
		if (!s_bn_ready) {
			s_bn_ready = true;
			BufNativo::Get().forget.store([](const std::vector<std::pair<uint64_t, uint64_t>>& chunks) { TextureOffload::MeshForget(chunks); },
			                              std::memory_order_release);
			const bool can = TextureOffload::Native() && TextureOffload::Ram18On();
			std::printf("[buf-nativo] %s\n", can ? "activo: memoria nativa y almacen RAM17/RAM18 disponibles (VRAM si mem1 etapa 2 esta encendida)"
			                                     : "INERTE: necesita la memoria nativa (KYTY_NATIVE_MEMORY / KYTY_RAIZ) y KYTY_RAM17=1; las "
			                                       "mallas solo se cuentan, siguen en la RAM");
			std::fflush(stdout);
		}
		static uint64_t s_bn_pending = 0;
		if (tick % 30 == 0) {
			// (buf7) Con take_mutex: el destructor 0xaa1d10 (OnMeshFree) no puede olvidar una malla entre Due y MeshTakeNow
			// (si no, sus trozos se tomaban despues de liberarse el bloque).
			std::lock_guard bn_take(BufNativo::Get().take_mutex);
			auto            due = BufNativo::Due(TextureOffload::RntMesh, &s_bn_pending);
			if (!due.empty()) {
				// Sus vigilantes de lectura puestos antes de que el hilo RAM18 los escriba y suelte (subidas pendientes antes).
				// (BUF-NATIVO fix2) ANTES de registrarlos: un toque entre MeshTakeNow y el vigilante quitaba un vigilante
				// que aun no estaba y el que se ponia despues quedaba huerfano (fallo de lectura sin fin, nadie lo quita).
				// Los rechazados devuelven el suyo.
				for (const auto& [begin, size, kind]: due) {
					AsyncUpload::DrainRange(begin, size);
					m_page_manager.UpdateReadWatchers<true>(begin, size);
				}
				const auto taken = TextureOffload::MeshTakeNow(due);
				for (const auto& [begin, size, kind]: due) {
					if (std::find(taken.begin(), taken.end(), std::pair<uint64_t, uint64_t> {begin, size}) == taken.end())
						m_page_manager.UpdateReadWatchers<false>(begin, size);
				}
			}
		}
		static auto s_bn_last = std::chrono::steady_clock::now();
		if (const auto now = std::chrono::steady_clock::now(); now - s_bn_last >= std::chrono::seconds(30)) {
			TextureOffload::MeshView v;
			if (TextureOffload::MeshRead(&v)) {
				s_bn_last          = now;
				auto&       r      = BufNativo::Get();
				const double MB    = 1024.0 * 1024.0;
				const auto  live   = BufNativo::LiveBytes();
				uint64_t    vused = 0, vsize = 0;
#if defined(_WIN32)
				TextureOffload::VramUse(&vused, &vsize);
#endif
				std::printf("[buf-nativo] 30 s: mallas vivas %.0f MB (%llu construidas, %llu destruidas; esperando %.0f MB); NATIVO fuera "
				            "de RAM %.0f MB (pico %.0f; tomadas %llu = %.0f MB, rechazadas %llu, escritas al almacen %.0f MB, almacen VRAM "
				            "%.0f/%.0f MB); fallos CPU juego %llu emulador %llu; restauraciones %llu (%.1f MB, %.1f ms max); olvidadas %llu "
				            "(%.0f MB); perdidas %llu, sin almacen %llu; CBuffer descartados %llu, cambiados %llu\n",
				            live / MB, static_cast<unsigned long long>(r.meshes.load()), static_cast<unsigned long long>(r.freed.load()),
				            s_bn_pending / MB, v.now / MB, v.peak / MB, static_cast<unsigned long long>(v.taken), v.taken_bytes / MB,
				            static_cast<unsigned long long>(v.refused), v.written_bytes / MB, vused / MB, vsize / MB,
				            static_cast<unsigned long long>(v.faults_game), static_cast<unsigned long long>(v.faults_emu),
				            static_cast<unsigned long long>(v.reads), v.read_bytes / MB, v.read_us_max / 1000.0,
				            static_cast<unsigned long long>(v.forgot), v.forgot_bytes / MB, static_cast<unsigned long long>(v.lost),
				            static_cast<unsigned long long>(v.failed), static_cast<unsigned long long>(r.rejected.load()),
				            static_cast<unsigned long long>(r.stale.load()));
				std::fflush(stdout);
			}
		}
	}
	if (m_graphics.CanReportMemoryUsage()) {
		// (vram-budget) On a small video memory heap the driver's budget and usage read again every 30 flips (~0.5 s; one
		// vkGetPhysicalDeviceMemoryProperties2): VMA reads them only every 30 of its own block allocations, so memory the
		// driver takes itself (pipelines, ~2.4 GB) and another program's share of the card showed late or never.
		if (m_graphics.small_vram_heap && tick % 30 == 0) m_graphics.RefreshMemoryBudget();
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	// (VRAM-PRES) KYTY_TEX_NATIVA_PRESUPUESTO=1 (por defecto). Log 20261124-044349: el recolector (vram8) solo pedia 512 MiB
	// libres bajo el presupuesto del driver (6998 MiB), no bajo el de [vram] (6123 = driver - 1/8): la VRAM se quedaba en
	// 6250-6800. Y la memoria de las texturas nativas (mips, pool, retirada) no se veia ni se devolvia. Aqui: cada pasada
	// se devuelve lo retirado; si la VRAM pasa del presupuesto menos PresMargin, el pool se vacia y no se rellena.
	constexpr uint64_t PresMargin = uint64_t {256} << 20u;
	const bool         pres       = SparsePresupuestoOn() && TexNativa::On() && m_graphics.CanReportMemoryUsage();
	uint64_t           pres_budget = 0;
	// (VRAM-DESG) KYTY_VRAM_DESG_GC=0: el objetivo de antes. Log 20261124-052155: la VRAM fija (bufferes 1539 MiB +
	// driver ~1330 + nativas ~640 + RT que no se expulsan) ya pasa de 6165 (presupuesto de [vram] = driver 7046 - 1/8):
	// el recolector perseguia un objetivo inalcanzable y expulsaba texturas de 5-10 s que volvian (16,8 GB en 8 min,
	// 2,4 GB/min). Ahora el objetivo es el presupuesto del driver menos DesgReserve (lo que de verdad no se debe pasar).
	static const bool desg_gc = [] { const char* v = std::getenv("KYTY_VRAM_DESG_GC"); return v == nullptr || *v != '0'; }();
	constexpr uint64_t DesgReserve = uint64_t {384} << 20u; // empieza a expulsar con menos hueco que esto
	constexpr uint64_t DesgHyst    = uint64_t {256} << 20u; // y sigue hasta tener esto mas (sin ir y venir)
	const bool         desg_on     = desg_gc && m_graphics.CanReportMemoryUsage();
	const uint64_t     desg_drv_room = desg_on ? m_graphics.GetVideoMemoryRoom() + m_graphics.GetVideoBlockSlack() : 0;
	// (VRAM-8GB, 11-24; solo con la VRAM del adaptador <= 8 GB = small_vram_heap; KYTY_VRAM_8GB=0 lo apaga) Log
	// 20261124-075619: VRAM 6422-6465 MiB con el presupuesto de [vram] en 6040 y 583 MiB aun bajo el del driver: el
	// recolector (que solo miraba el del driver menos 384) no expulsaba nada (evicted 0). Ahora el hueco que cuenta es el
	// menor de los dos: el del driver y el de [vram] (GetTotalMemoryBudget). Se siguen expulsando solo imagenes sin usar
	// >= 30 s (lo cercano se usa cada fotograma y no se toca).
	static const bool vram_8gb = [] { const char* v = std::getenv("KYTY_VRAM_8GB"); return v == nullptr || *v != '0'; }();
	uint64_t desg_room = desg_drv_room;
	if (desg_on && vram_8gb && m_graphics.small_vram_heap && !VramTope::On()) { // (VRAM-TOPE) el tope manda
		const uint64_t b8 = m_graphics.GetTotalMemoryBudget(false);
		const uint64_t u8 = m_graphics.GetDeviceMemoryUsage();
		desg_room = std::min(desg_room, (b8 > u8 ? b8 - u8 : 0) + m_graphics.GetVideoBlockSlack());
	}
	// (VRAM-DRV) Techo propio: el uso REAL del proceso (GetDeviceMemoryUsage: VMA + fuera de VMA + driver) contra el
	// TechoPct % del presupuesto de [vram]. Log 055740: el proceso estuvo por encima del presupuesto casi toda la partida
	// (5430-6177 de 5394) con el recolector limitado a lo de >= 600-1800 pasadas sin usar: lo que faltaba no se podia
	// sacar y el driver acababa moviendo memoria en uso. Lo que pase del techo cuenta como hueco que falta.
	// (VRAM-DRV2) Con histeresis (ver vram-drv.h): activo desde TechoPct hasta TechoBajoPct; drv_need = lo que falta
	// hasta TechoBajoPct, drv_excess = lo que pasa de TechoPct (decide la edad minima).
	uint64_t       drv_need      = 0;
	uint64_t       drv_excess    = 0;
	const uint64_t desg_room_leg = desg_room; // el hueco de antes, sin el techo propio
	if (desg_on && VramDrv::On()) {
		const uint64_t bt     = m_graphics.GetTotalMemoryBudget(false);
		const uint64_t ut     = m_graphics.GetDeviceMemoryUsage();
		// (VRAM-TOPE) Con el tope unico, el nivel 3 (expulsion) empieza al 98 % del tope efectivo y sigue hasta el 93 %.
		const uint64_t tope_b = VramTope::On() ? VramTope::g_tope_bytes.load(std::memory_order_relaxed) : 0;
		const uint64_t alto   = tope_b != 0 ? tope_b / 100 * VramTope::N3Pct : bt / 100 * VramDrv::TechoPct;
		const uint64_t bajo   = tope_b != 0 ? tope_b / 100 * VramTope::N3LowPct : bt / 100 * VramDrv::TechoBajoPct;
		bool           activo = VramDrv::g_activo.load(std::memory_order_relaxed);
		if (!activo && ut > alto) {
			activo = true;
			VramDrv::g_activaciones.fetch_add(1, std::memory_order_relaxed);
		} else if (activo && ut <= bajo) {
			activo = false;
		}
		// (TOPE2) Con el tope, el nivel 3 solo si vram-tope.h lo permite (nivel 2 agotado y no bloqueado por inutil).
		if (tope_b != 0 && !VramTope::g_n3_permitido.load(std::memory_order_relaxed)) activo = false;
		VramDrv::g_activo.store(activo, std::memory_order_relaxed);
		if (activo && ut > bajo) {
			drv_need   = ut - bajo;
			drv_excess = ut > alto ? ut - alto : 0;
			desg_room  = 0;
		}
	}
	// (MALLA-NATIVA fase 1h) Lo que las mallas no pudieron reservar cuenta como hueco que falta: el recolector expulsa
	// texturas viejas (>= 30 s sin usar) por ello en vez de dejar la malla en el camino normal.
	if (desg_on) {
		const uint64_t malla_want = MallaNativa::TakeBudgetWant();
		desg_room                 = desg_room > malla_want ? desg_room - malla_want : 0;
	}
	if (pres) {
		pres_budget             = m_graphics.GetTotalMemoryBudget(false);
		const bool     pressure = desg_on ? desg_room < DesgReserve + DesgHyst // (VRAM-DESG) bajo el del driver
		                                  : m_total_used_memory + PresMargin > pres_budget;
		const uint64_t freed    = SparsePresupuestoReclaim(m_graphics, pressure);
		m_pres_freed_bytes += freed;
		// (W14, SPARSE-COMPACTAR) Cada 30 pasadas: los mips de las nativas sin usar en >= 120 fotogramas pueden pasar de
		// los bloques poco llenos a los llenos (ver sparseImage.cpp).
		if (static uint64_t compactar_tick = 0; m_gc_tick >= compactar_tick + 30) {
			compactar_tick = m_gc_tick;
			thread_local std::vector<VulkanImage*> sin_uso;
			sin_uso.clear();
			m_slot_images.ForEach([&](ImageId, Image& image) {
				if (image.registered && image.IsSparse() && SparseIsPerLevel(image.backing) &&
				    m_lru_cache.TickOf(image.lru_id) + 120 <= m_gc_tick)
					sin_uso.push_back(&image.backing);
			});
			SparseCompactar(m_graphics, m_scheduler, sin_uso);
		}
		if (freed != 0) m_total_used_memory -= std::min(m_total_used_memory, freed);
		static auto last_pres = std::chrono::steady_clock::now();
		if (const auto now = std::chrono::steady_clock::now(); now - last_pres >= std::chrono::seconds(30)) {
			last_pres     = now;
			const auto st = SparseNativeStatsNow();
			std::printf("[vram-pres] 30 s: VRAM usada %llu / presupuesto %llu MiB; nativa %llu MB (mips ligados %llu, retirada "
			            "pendiente %llu, pool %llu); devuelto por presupuesto %llu MB; expulsiones nativas %llu (%llu MB), "
			            "expulsiones totales %llu (%llu MiB)\n",
			            static_cast<unsigned long long>(m_total_used_memory >> 20u),
			            static_cast<unsigned long long>(pres_budget >> 20u),
			            static_cast<unsigned long long>((st.bound + st.retired + st.pool) >> 20u),
			            static_cast<unsigned long long>(st.bound >> 20u), static_cast<unsigned long long>(st.retired >> 20u),
			            static_cast<unsigned long long>(st.pool >> 20u),
			            static_cast<unsigned long long>(m_pres_freed_bytes >> 20u),
			            static_cast<unsigned long long>(m_pres_native_evicted),
			            static_cast<unsigned long long>(m_pres_native_evicted_bytes >> 20u),
			            static_cast<unsigned long long>(VramSaverGet(VramEvicted)),
			            static_cast<unsigned long long>(VramSaverGet(VramEvictedBytes) >> 20u));
			std::fflush(stdout);
		}
	}
	// (VRAM-JUEGO) KYTY_VRAM_JUEGO=1 (por defecto): cada ~2 s, si la VRAM pasa del presupuesto, se baja el pool de streaming
	// del propio juego (0x2f69898, leido por su gestor 0xac9ea0) y el juego suelta mips (op 2 -> desligado sparse).
	// (VRAM-8GB) Las lejanas ("Cache Unused Textures"=0, pasada 1 de 0xac9d80) iban dentro de VramJuego::On() (apagado
	// por defecto) y nunca se ejecutaban (el log 075619 no tiene ninguna linea [vram-lejanas]). Ahora corren solas con
	// la VRAM del adaptador <= 8 GB; el pool del juego (0x2f69898) no se toca.
	if (!VramTope::On() && (VramJuego::On() || (m_graphics.small_vram_heap && VramJuego::LejanasOn())) &&
	    m_graphics.CanReportMemoryUsage()) { // (VRAM-TOPE) el pool lo lleva vram-tope.h
		static auto last_vj = std::chrono::steady_clock::now();
		if (const auto now = std::chrono::steady_clock::now(); now - last_vj >= std::chrono::seconds(2)) {
			last_vj = now;
			VramJuego::Tick(m_graphics.GetDeviceMemoryUsage(), m_graphics.GetTotalMemoryBudget(false));
		}
	}
	// (VRAM-TECHO) KYTY_VRAM_TECHO=1 (por defecto): cada ~1 s; por encima del 90 % del presupuesto el propio juego baja la
	// resolucion de las texturas lejanas (pasada 1 de 0xac9d80; ver vram-juego.h). Log [vram-techo] cada 5 s.
	if (VramJuego::TechoOn() && m_graphics.CanReportMemoryUsage()) {
		static auto last_vt = std::chrono::steady_clock::now();
		static const auto t0_vt = last_vt;
		if (const auto now = std::chrono::steady_clock::now(); now - last_vt >= std::chrono::seconds(1)) {
			last_vt = now;
			// (VRAM-TOPE) Tope unico: tarjeta (tabla del usuario) y presupuesto del driver para el proceso (uso + hueco).
			const uint64_t uso_t  = m_graphics.GetDeviceMemoryUsage();
			const uint64_t room_t = m_graphics.GetVideoMemoryRoom(); // (N3-REAL) tambien guarda el presupuesto real
			const uint64_t drv_t  = VramDrv::g_drv_budget_bytes.load(std::memory_order_relaxed);
			const uint64_t tope_t =
			    VramTope::Tick(VramCardBytes(), uso_t, drv_t != 0 ? drv_t : uso_t + room_t,
			                   static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - t0_vt).count()),
			                   tick, VramSaverGet(VramEvictedBytes), TexNativa::c_freed_bytes.load(std::memory_order_relaxed),
			                   VramDrv::g_activo.load(std::memory_order_relaxed),
			                   VramDrv::g_reup_bytes.load(std::memory_order_relaxed)); // (TOPE2) re-subido
			// Nivel 1 (VRAM_TECHO): sus umbrales 90/80 % se aplican sobre el tope efectivo.
			VramJuego::TickTecho(
			    uso_t, tope_t != 0 ? tope_t : m_graphics.GetTotalMemoryBudget(false),
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - t0_vt).count()),
			    TexNativa::c_levels_unbound.load(std::memory_order_relaxed),
			    TexNativa::c_freed_bytes.load(std::memory_order_relaxed), SparseNativeStatsNow().bound);
		}
	}
	// (VRAM-DRV) Desglose medido de la VRAM del proceso cada 5 s ([vram-drv]).
	if (VramDrv::On() && m_graphics.CanReportMemoryUsage()) {
		static auto last_vd = std::chrono::steady_clock::now();
		if (const auto now = std::chrono::steady_clock::now(); now - last_vd >= std::chrono::seconds(5)) {
			last_vd = now;
			VramDrvReport(m_graphics);
			std::printf("[vram-drv] recolector con techo propio: pasadas %llu, lo que faltaba como mucho %llu MiB; "
			            "expulsadas en total %llu (%llu MiB); nunca lo usado en los ultimos %llu fotogramas\n",
			            static_cast<unsigned long long>(VramDrv::g_gc_runs.load()),
			            static_cast<unsigned long long>(VramDrv::g_gc_need_max.load() >> 20u),
			            static_cast<unsigned long long>(VramSaverGet(VramEvicted)),
			            static_cast<unsigned long long>(VramSaverGet(VramEvictedBytes) >> 20u),
			            static_cast<unsigned long long>(VramDrv::InUseTicks));
			std::printf("[vram-drv] re-subidas tras expulsion %llu (%llu MiB); techo %s (%llu-%llu %%, activado %llu veces); "
			            "no expulsadas por re-subida reciente (<%llu s) %llu; pasadas cortadas por %llu MiB/s %llu\n",
			            static_cast<unsigned long long>(VramDrv::g_reup_n.load()),
			            static_cast<unsigned long long>(VramDrv::g_reup_bytes.load() >> 20u),
			            VramDrv::g_activo.load() ? "ACTIVO" : "en reposo",
			            static_cast<unsigned long long>(VramDrv::TechoBajoPct),
			            static_cast<unsigned long long>(VramDrv::TechoPct),
			            static_cast<unsigned long long>(VramDrv::g_activaciones.load()),
			            static_cast<unsigned long long>(VramDrv::ReupProtegeMs / 1000),
			            static_cast<unsigned long long>(VramDrv::g_skip_reup.load()),
			            static_cast<unsigned long long>(VramDrv::MaxMiBps),
			            static_cast<unsigned long long>(VramDrv::g_cut_rate.load()));
			std::printf("[vram-reposo] pasadas vacias %llu -> reposo de %llu fotogramas; pasadas ahorradas %llu\n",
			            static_cast<unsigned long long>(VramDrv::g_reposo_vacias.load()),
			            static_cast<unsigned long long>(VramDrv::ReposoTicks),
			            static_cast<unsigned long long>(VramDrv::g_reposo_saltadas.load()));
			std::fflush(stdout);
		}
	}
	// KYTY_VRAM_SAVER: where the video memory stands every 2 s (a GPU short of it only).
	if (m_graphics.VramSaver()) {
		static auto last_vram_log = std::chrono::steady_clock::now();
		if (const auto now = std::chrono::steady_clock::now(); now - last_vram_log >= std::chrono::seconds(2)) {
			last_vram_log = now;
			std::printf("[vram] used %llu / budget %llu MiB, evicted %llu (%llu MiB), fallback host %llu, failed %llu, "
			            "rescued %llu\n",
			            static_cast<unsigned long long>(m_graphics.GetDeviceMemoryUsage() >> 20u),
			            static_cast<unsigned long long>(m_graphics.GetTotalMemoryBudget(false) >> 20u),
			            static_cast<unsigned long long>(VramSaverGet(VramEvicted)),
			            static_cast<unsigned long long>(VramSaverGet(VramEvictedBytes) >> 20u),
			            static_cast<unsigned long long>(VramSaverGet(VramFallbackHost)),
			            static_cast<unsigned long long>(VramSaverGet(VramFailed)),
			            static_cast<unsigned long long>(VramSaverGet(VramRescued)));
			std::fflush(stdout);
		}
		// (GC-GPU-SUCIA) Cada 30 s: imagenes pintadas por la GPU borradas sin descarga (por motivo) y retenidas.
		static auto last_gcs = std::chrono::steady_clock::now();
		if (const auto now = std::chrono::steady_clock::now(); now - last_gcs >= std::chrono::seconds(30)) {
			last_gcs = now;
			std::printf("[gc-gpu-sucia] (KYTY_GC_GPU_SUCIA=%d) pintadas por la GPU borradas sin descarga: por escritura de la CPU %llu "
			            "(%llu MiB; con el arreglo debe quedar en 0), por buffer de la GPU %llu (%llu MiB, correcto); retenidas por el "
			            "arreglo %llu veces (%llu MiB)\n",
			            GcGpuSucia() ? 1 : 0, static_cast<unsigned long long>(g_gcs_cpu.load()),
			            static_cast<unsigned long long>(g_gcs_cpu_bytes.load() >> 20u), static_cast<unsigned long long>(g_gcs_buf.load()),
			            static_cast<unsigned long long>(g_gcs_buf_bytes.load() >> 20u), static_cast<unsigned long long>(g_gcs_ret.load()),
			            static_cast<unsigned long long>(g_gcs_ret_bytes.load() >> 20u));
			std::fflush(stdout);
		}
		// (vram8) Every 60 s: the array images of 256 MiB or more made, the BC pool views kept to their layer bound
		// (KYTY_VRAM8_BC_BOUND) and the video memory evicted in that minute.
		static auto     last_vram8      = std::chrono::steady_clock::now();
		static uint64_t last_vram8_evic = 0;
		if (const auto now = std::chrono::steady_clock::now(); now - last_vram8 >= std::chrono::seconds(60)) {
			const double   minutes = std::chrono::duration<double>(now - last_vram8).count() / 60.0;
			const uint64_t evic    = VramSaverGet(VramEvictedBytes);
			last_vram8             = now;
			std::printf("[vram8] arrays grandes: %llu (%llu MB creados) / vistas BC sin ensanchar %llu veces (%llu MB evitados), "
			            "expulsiones por minuto %.0f MiB / recolector: %llu pasadas evitadas con hueco, %llu hechas (KYTY_VRAM8_GC)\n",
			            static_cast<unsigned long long>(Vram8::big_n.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(Vram8::big_bytes.load(std::memory_order_relaxed) >> 20u),
			            static_cast<unsigned long long>(Vram8::kept_n.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(Vram8::kept_bytes.load(std::memory_order_relaxed) >> 20u),
			            minutes > 0.0 ? static_cast<double>((evic - last_vram8_evic) >> 20u) / minutes : 0.0,
			            static_cast<unsigned long long>(Vram8::gc_skipped.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(Vram8::gc_runs.load(std::memory_order_relaxed)));
			std::fflush(stdout);
			last_vram8_evic = evic;
		}
	}
	// (VRAM-REPARTO) Cada 30 s: expulsiones/s, MB re-subidos/s y el presupuesto efectivo (el del driver y el de las caches).
	if (m_graphics.CanReportMemoryUsage()) {
		static auto     last_rep = std::chrono::steady_clock::now();
		static uint64_t last_ev = 0, last_rb = 0, last_rn = 0;
		if (const auto now = std::chrono::steady_clock::now(); now - last_rep >= std::chrono::seconds(30)) {
			const double   secs = std::chrono::duration<double>(now - last_rep).count();
			last_rep            = now;
			const uint64_t ev   = VramSaverGet(VramEvicted);
			const uint64_t rb   = VramReparto::reup_bytes.load(std::memory_order_relaxed);
			const uint64_t rn   = VramReparto::reup_n.load(std::memory_order_relaxed);
			std::printf("[vram-reparto] 30 s: expulsiones %.2f/s, re-subidas %.2f/s (%.1f MB/s); presupuesto efectivo %llu MiB "
			            "(caches %llu), VRAM usada %llu, hueco bajo el driver %llu MiB\n",
			            static_cast<double>(ev - last_ev) / secs, static_cast<double>(rn - last_rn) / secs,
			            static_cast<double>((rb - last_rb) >> 10u) / 1024.0 / secs,
			            static_cast<unsigned long long>(m_graphics.GetTotalMemoryBudget(false) >> 20u),
			            static_cast<unsigned long long>(m_graphics.GetTotalMemoryBudget() >> 20u),
			            static_cast<unsigned long long>(m_graphics.GetDeviceMemoryUsage() >> 20u),
			            static_cast<unsigned long long>(m_graphics.GetVideoMemoryRoom() >> 20u));
			std::fflush(stdout);
			last_ev = ev; last_rb = rb; last_rn = rn;
		}
	}
	// (R2-INTERMEDIAS, 11-25) KYTY_VRAM_INTERMEDIAS=1 (apagado por defecto; lo ponen los cmd 50/51 de 1080p y 1440p): a los
	// 60 s del arranque y luego cada 60 s, la VRAM real (VMA; sparse: lo ligado) de todas las imagenes intermedias -
	// render targets, profundidad y storage escritas por la GPU, sin la salida de video - sumada por tamano, formato y uso.
	// Para comparar 1080p con 1440p con el mismo minuto en Boletaria. Solo lee (m_lock ya tomado aqui); nada cambia.
	{
		static const bool inter_on = [] {
			const char* v  = std::getenv("KYTY_VRAM_INTERMEDIAS");
			const bool  on = v != nullptr && v[0] == '1';
			if (on) {
				std::printf("[vram-intermedias] KYTY_VRAM_INTERMEDIAS=1: a los 60 s (y cada 60 s) VRAM de RT/profundidad/storage por "
				            "tamano y formato; KYTY_RENDER_HEIGHT=%s\n",
				            std::getenv("KYTY_RENDER_HEIGHT") != nullptr ? std::getenv("KYTY_RENDER_HEIGHT") : "(sin poner)");
				std::fflush(stdout);
			}
			return on;
		}();
		static const auto inter_start = std::chrono::steady_clock::now();
		static auto       inter_last  = inter_start;
		const auto        inter_now   = std::chrono::steady_clock::now();
		if (inter_on && inter_now - inter_last >= std::chrono::seconds(60)) {
			inter_last = inter_now;
			struct Grupo {
				uint64_t bytes = 0, n = 0;
			};
			// clave: ancho, alto, formato de Vulkan, formato del juego, capas, niveles, uso (r/d/s)
			std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::string>, Grupo> grupos;
			uint64_t total = 0, total_n = 0, por_uso[3] {}, sin_memoria_propia = 0;
			m_slot_images.ForEach([&](ImageId, const Image& image) {
				if (!image.registered || image.usage.video_out) return;
				const auto& u     = image.usage;
				const bool  depth = u.depth_target || image.info.IsDepth();
				if (!(u.render_target || depth || u.storage)) return;
				uint64_t b = 0;
				if (image.backing.allocation != nullptr) {
					VmaAllocationInfo ai {};
					vmaGetAllocationInfo(m_graphics.allocator, image.backing.allocation, &ai);
					b = ai.size;
				} else if (image.IsSparse()) {
					b = SparseBoundBytes(image.backing);
				}
				if (b == 0) sin_memoria_propia++; // (alias de memoria de otra imagen, o sin memoria aun)
				std::string uso;
				if (u.render_target) uso += 'r';
				if (depth) uso += 'd';
				if (u.storage) uso += 's';
				auto& g = grupos[{image.info.extent.width, image.info.extent.height, static_cast<uint32_t>(image.info.pixel_format),
				                  static_cast<uint32_t>(image.info.guest_format), image.info.resources.layers,
				                  image.info.resources.levels, uso}];
				g.bytes += b;
				g.n++;
				total += b;
				total_n++;
				por_uso[depth ? 1 : (u.render_target ? 0 : 2)] += b;
			});
			std::vector<std::pair<uint64_t, std::string>> lista;
			for (const auto& [k, g]: grupos) {
				char item[160];
				std::snprintf(item, sizeof(item), "%ux%u vk%u g%u c%u n%u %s x%llu=%.1f", std::get<0>(k), std::get<1>(k), std::get<2>(k),
				              std::get<3>(k), std::get<4>(k), std::get<5>(k), std::get<6>(k).c_str(),
				              static_cast<unsigned long long>(g.n), static_cast<double>(g.bytes) / 1048576.0);
				lista.emplace_back(g.bytes, item);
			}
			std::sort(lista.begin(), lista.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
			const double secs = std::chrono::duration<double>(inter_now - inter_start).count();
			std::printf("[vram-intermedias] %.0f s, render %s: %llu imagenes, VRAM %.1f MiB = RT %.1f + profundidad %.1f + storage %.1f "
			            "(sin memoria propia %llu); %zu grupos (WxH vkFormato gFormatoJuego capas niveles uso xN=MiB), de mayor a menor:",
			            secs, std::getenv("KYTY_RENDER_HEIGHT") != nullptr ? std::getenv("KYTY_RENDER_HEIGHT") : "?",
			            static_cast<unsigned long long>(total_n), static_cast<double>(total) / 1048576.0,
			            static_cast<double>(por_uso[0]) / 1048576.0, static_cast<double>(por_uso[1]) / 1048576.0,
			            static_cast<double>(por_uso[2]) / 1048576.0, static_cast<unsigned long long>(sin_memoria_propia), lista.size());
			for (const auto& [b, s]: lista) std::printf(" %s;", s.c_str());
			std::printf("\n");
			std::fflush(stdout);
		}
	}
	// Every 30 seconds: where the video memory and the process's memory stand (the run log).
	static auto last_report = std::chrono::steady_clock::now();
	if (m_graphics.CanReportMemoryUsage() && std::chrono::steady_clock::now() - last_report >= std::chrono::seconds(30)) {
		last_report = std::chrono::steady_clock::now();
		const auto [private_bytes, working_set] = LocalPlatform::ProcessMemory();
		std::printf("Video memory: %llu MiB in use of a %llu MiB budget (collecting from %llu). System memory: the process "
		            "holds %llu MiB (%llu MiB resident), %llu MiB of it Vulkan allocations in system RAM\n",
		            static_cast<unsigned long long>(m_total_used_memory >> 20u),
		            static_cast<unsigned long long>(m_graphics.GetTotalMemoryBudget() >> 20u),
		            static_cast<unsigned long long>(m_trigger_gc_memory >> 20u),
		            static_cast<unsigned long long>(private_bytes >> 20u), static_cast<unsigned long long>(working_set >> 20u),
		            static_cast<unsigned long long>(m_graphics.GetHostMemoryUsage() >> 20u));
		// Where the video memory goes (VMA's allocations by memory kind, the driver's own) and the buffer cache's
		// share: on a 16 GB card the use climbed past the 10 GB budget while the texture collector deleted images.
		std::printf("Buffer cache: %llu MiB registered\n", static_cast<unsigned long long>(m_buffer_cache.CacheBytes() >> 20u));
		// RAM diet: the emulator's own host-visible rings (driver memory in the process's private commit).
		std::printf("RAM diet (30 s): image upload ring %llu MiB (KYTY_STAGING_RING_MB), Vulkan system RAM %llu MiB\n",
		            static_cast<unsigned long long>(BufferCacheStagingRingSize() >> 20u),
		            static_cast<unsigned long long>(m_graphics.GetHostMemoryUsage() >> 20u));
		// The commit side (16 GB PCs rebooted at the commit limit, not at full RAM): the system's charge and limit,
		// and the guest backing store's share of it (0 when it is file-backed: KYTY_BACKING_FILE).
		{
			const auto commit  = LocalPlatform::SystemCommit();
			const char* file   = std::getenv("KYTY_BACKING_FILE");
			const char* demand = std::getenv("KYTY_BACKING_ON_DEMAND");
			const bool  in_file = file != nullptr && file[0] != '\0' && !(file[0] == '0' && file[1] == '\0');
			std::printf("Commit: system %llu of %llu MiB; emulator private %llu MiB, working set %llu MiB, Vulkan system "
			            "RAM %llu MiB, buffer cache %llu MiB; guest backing %s (13824 MiB store)\n",
			            static_cast<unsigned long long>(commit.total >> 20u),
			            static_cast<unsigned long long>(commit.limit >> 20u),
			            static_cast<unsigned long long>(private_bytes >> 20u), static_cast<unsigned long long>(working_set >> 20u),
			            static_cast<unsigned long long>(m_graphics.GetHostMemoryUsage() >> 20u),
			            static_cast<unsigned long long>(m_buffer_cache.CacheBytes() >> 20u),
			            in_file ? "file (see the Guest memory line at launch)"
			            : GuestLazyCommit::Enabled() ? "page file, committed at first touch (KYTY_GUEST_LAZY_COMMIT)"
			            : (demand != nullptr && demand[0] == '1') ? "page file, on demand" : "page file, all committed");
			// KYTY_GUEST_LAZY_COMMIT: the store's committed share, its commit calls and their time.
			if (const auto* fresh = FreshMemory::Get())
				GuestLazyCommit::Report(fresh->faults.load() + fresh->marked.load(), fresh->watched_bytes.load());
			else GuestLazyCommit::Report(0, 0);
		}
		m_graphics.LogVideoMemory("30 s");
		// (RT-ALIAS-MEM) Lo ahorrado por compartir memoria entre RT que el juego solapa en sus arenas.
		if (RtAliasMemOn()) {
			const auto ma = Image::MemAliasCounters();
			std::printf("[rt-alias-mem] KYTY_RT_ALIAS_MEM=1: ahorrado ahora %.1f MiB de VRAM (%llu RT de las arenas sin "
			            "VkDeviceMemory propia, ligados a la de otro que los contiene); acumulado: ligados %llu, rechazados %llu "
			            "(tipo de memoria, tamano o alineacion), reinicios a UNDEFINED por pisado %llu, leidos tras pisarlos "
			            "%llu (recreados con memoria propia)\n",
			            static_cast<double>(ma.ahora_bytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(ma.ahora_n),
			            static_cast<unsigned long long>(ma.unidas), static_cast<unsigned long long>(ma.rechazadas),
			            static_cast<unsigned long long>(ma.reinicios), static_cast<unsigned long long>(ma.malas));
		}
		// (VRAM-DESG) KYTY_VRAM_DESG=0 lo apaga. El desglose de "images and other device-local" (un recorrido de las
		// ~1700 imagenes cada 30 s, m_lock ya tomado): memoria real de cada imagen (VMA; las nativas, lo ligado a sus
		// mips, que el driver cuenta fuera de VMA) por tipo; lo que lleva >30 s sin usarse; las que empiezan en la misma
		// direccion que otra (alias/formatos: copia doble); las arenas RT del juego (Ghidra 0x91c680, llamada desde
		// 0x90ad80 con el modo 4: RTSharedAllocator 0x2af40000 = 687,2 MiB, RTPersistAllocator 0x38170000 = 897,4 MiB)
		// con su rango solapado; y los render targets de mas de 1920x1080 (la vista de salida del juego, 0x90ad80).
		static const bool vram_desg = [] { const char* v = std::getenv("KYTY_VRAM_DESG"); return v == nullptr || *v != '0'; }();
		if (vram_desg) {
			enum { Tex, Nat, RtShared, RtPersist, RtOther, GpuImg, Parked, Cats };
			static const char* const names[Cats] = {"texturas", "nativas", "RTShared", "RTPersist", "RT/profundidad fuera de arenas",
			                                         "imagenes escritas por la GPU", "aparcadas/sin registrar"};
			uint64_t bytes[Cats] {}, count[Cats] {}, stale[Cats] {};
			uint64_t dup_n = 0, dup_bytes = 0, big_rt_n = 0, big_rt_bytes = 0;
			// (RT-LISTA, 11-25) Cada render target grande o fuera de arenas, uno a uno (log 104537: +5 RT > 1920x1080, +230
			// MiB, y RT fuera de arenas 7 -> 11, +170 MiB, sin saber cuales): direccion, tamano, formato, capas, uso.
			struct RtItem { uint64_t addr, bytes, age; uint32_t w, h, fmt, layers, levels; int arena; char use[6]; bool big; };
			std::vector<RtItem> rt_items;
			std::vector<std::pair<uint64_t, uint64_t>> arena_spans[2];
			std::unordered_map<uint64_t, uint32_t> starts;
			starts.reserve(4096);
			const uint64_t now_tick = m_gc_tick;
			m_slot_images.ForEach([&](ImageId, const Image& image) {
				uint64_t b = 0;
				if (image.backing.allocation != nullptr) {
					VmaAllocationInfo ai {};
					vmaGetAllocationInfo(m_graphics.allocator, image.backing.allocation, &ai);
					b = ai.size;
				} else if (image.IsSparse()) {
					b = SparseBoundBytes(image.backing);
				}
				int cat = Tex;
				const auto& u = image.usage;
				const int   arena = image.info.data.Empty() ? 0
				                    : Loader::DemonsSoulsMemMap::RtArenaOf(image.info.data.address, image.info.data.End());
				if (!image.registered) cat = Parked;
				else if (image.IsSparse()) cat = Nat;
				else if (arena == 1) cat = RtShared;
				else if (arena == 2) cat = RtPersist;
				else if (u.render_target || u.depth_target || u.video_out) cat = RtOther;
				else if (u.storage || image.IsGpuModified()) cat = GpuImg;
				bytes[cat] += b;
				count[cat]++;
				if (image.registered) {
					if (m_lru_cache.TickOf(image.lru_id) + 1800 < now_tick) stale[cat] += b;
					if (!image.info.data.Empty() && starts[image.info.data.address]++ != 0) { dup_n++; dup_bytes += b; }
					if (arena != 0) arena_spans[arena - 1].emplace_back(image.info.data.address, image.info.data.End());
					const bool big = (u.render_target || u.depth_target) &&
					                 (image.info.extent.width > 1920 || image.info.extent.height > 1080);
					if (big) {
						big_rt_n++;
						big_rt_bytes += b;
					}
					if ((big || cat == RtOther) && b >= (uint64_t {4} << 20u) && rt_items.size() < 32) {
						RtItem it {image.info.data.address, b, now_tick - std::min(now_tick, m_lru_cache.TickOf(image.lru_id)),
						           image.info.extent.width, image.info.extent.height,
						           static_cast<uint32_t>(image.info.pixel_format), image.info.resources.layers,
						           image.info.resources.levels, arena, {}, big};
						int k = 0;
						if (u.render_target) it.use[k++] = 'r';
						if (u.depth_target || image.info.IsDepth()) it.use[k++] = 'd';
						if (u.storage) it.use[k++] = 's';
						if (u.video_out) it.use[k++] = 'v';
						if (u.texture) it.use[k++] = 't';
						it.use[k] = 0;
						rt_items.push_back(it);
					}
				}
			});
			std::printf("[vram-desg] imagenes (MiB, n, sin usar >30 s MiB):");
			uint64_t all = 0;
			for (int c = 0; c < Cats; c++) {
				all += bytes[c];
				std::printf(" %s %llu (%llu, %llu);", names[c], static_cast<unsigned long long>(bytes[c] >> 20u),
				            static_cast<unsigned long long>(count[c]), static_cast<unsigned long long>(stale[c] >> 20u));
			}
			std::printf(" total %llu\n", static_cast<unsigned long long>(all >> 20u));
			// (TOPE2) Reparto para la linea [vram-tope].
			VramTope::g_rep_tex.store(bytes[Tex] + bytes[Nat], std::memory_order_relaxed);
			VramTope::g_rep_rt.store(bytes[RtShared] + bytes[RtPersist] + bytes[RtOther], std::memory_order_relaxed);
			VramTope::g_rep_gpu.store(bytes[GpuImg], std::memory_order_relaxed);
			VramTope::g_rep_otras.store(bytes[Parked], std::memory_order_relaxed);
			VramTope::g_rep_buf.store(m_buffer_cache.CacheBytes(), std::memory_order_relaxed);
			uint64_t stale_all = 0;
			for (int c = 0; c < Cats; c++) stale_all += stale[c];
			VramTope::g_rep_stale.store(stale_all, std::memory_order_relaxed);
			for (int a = 0; a < 2; a++) {
				auto& s = arena_spans[a];
				std::sort(s.begin(), s.end());
				uint64_t sum = 0, uni = 0, cb = 0, ce = 0;
				for (const auto& [sb, se]: s) {
					sum += se - sb;
					if (sb >= ce) { uni += ce - cb; cb = sb; ce = se; }
					else if (se > ce) ce = se;
				}
				uni += ce - cb;
				std::printf("[vram-desg] arena %s: rango del juego cubierto %llu MiB, rango sumado de sus imagenes %llu MiB "
				            "(solapado %llu MiB: imagenes distintas sobre la misma memoria)\n",
				            a == 0 ? "RTSharedAllocator (687 MiB)" : "RTPersistAllocator (897 MiB)",
				            static_cast<unsigned long long>(uni >> 20u), static_cast<unsigned long long>(sum >> 20u),
				            static_cast<unsigned long long>((sum - std::min(sum, uni)) >> 20u));
			}
			if (!rt_items.empty()) {
				std::sort(rt_items.begin(), rt_items.end(), [](const RtItem& x, const RtItem& y) { return x.addr < y.addr; });
				std::printf("[vram-desg] RT grandes (>1920x1080, G) y fuera de arenas (F), >= 4 MiB: direccion WxH fmt capas/niveles MiB uso "
				            "arena edad:");
				for (const auto& it: rt_items)
					std::printf(" %c0x%llx %ux%u f%u %u/%u %llu %s a%d e%llu;", it.big ? 'G' : 'F',
					            static_cast<unsigned long long>(it.addr), it.w, it.h, it.fmt, it.layers, it.levels,
					            static_cast<unsigned long long>(it.bytes >> 20u), it.use, it.arena,
					            static_cast<unsigned long long>(it.age));
				std::printf("\n");
			}
			std::printf("[vram-desg] duplicadas (misma direccion que otra imagen) %llu, %llu MiB; render targets de mas de "
			            "1920x1080 %llu, %llu MiB; bufferes de la cache %llu MiB\n",
			            static_cast<unsigned long long>(dup_n), static_cast<unsigned long long>(dup_bytes >> 20u),
			            static_cast<unsigned long long>(big_rt_n), static_cast<unsigned long long>(big_rt_bytes >> 20u),
			            static_cast<unsigned long long>(m_buffer_cache.CacheBytes() >> 20u));
		}
		std::fflush(stdout);
	}
	if (KytyGcLiveBudget() && m_graphics.CanReportMemoryUsage()) {
		static auto   kyty_budget_window = std::chrono::steady_clock::time_point {};
		const auto    now                = std::chrono::steady_clock::now();
		const uint64_t budget            = KytyGcLiveBudgetValue(m_graphics);
		if (now - kyty_budget_window >= std::chrono::seconds(2)) {
			kyty_budget_window = now;
			uint64_t trigger = 0, pressure = 0, critical = 0;
			KytyTextureGcThresholds(budget, trigger, pressure, critical);
			const auto moved = [](uint64_t a, uint64_t b) { return (a > b ? a - b : b - a) >= (uint64_t {64} << 20u); };
			if (moved(trigger, m_trigger_gc_memory)) {
				std::printf("KYTYGCBUDGET texture budget=%lluMB trigger=%lluMB pressure=%lluMB critical=%lluMB "
				            "used=%lluMB\n",
				            static_cast<unsigned long long>(budget >> 20), static_cast<unsigned long long>(trigger >> 20),
				            static_cast<unsigned long long>(pressure >> 20),
				            static_cast<unsigned long long>(critical >> 20),
				            static_cast<unsigned long long>(m_total_used_memory >> 20));
				std::fflush(stdout);
			}
			m_trigger_gc_memory = trigger;
		}
	} else if (m_graphics.CanReportMemoryUsage()) {
		// The budget Windows gives now, not the one at start-up: another program's video memory lowers it (a Discord
		// stream: 15.2 -> 13.9 GB on a 16 GB GPU), and past it Windows moves the game's memory out of video memory
		// (60 -> 20-30 fps). Its lowest of the last 30 s (KytyGcLiveBudgetValue), less GetTotalMemoryBudget's headroom
		// as at start-up. Below 12 GiB of budget it only lowers the start-up trigger (issue #276: an 8 GB RX 6650 XT's
		// budget went from 7378 to ~5570 MiB in the world; the collector, still at 6455, let the use stay up to 1.2 GiB
		// over it once frame generation took its share, and frames waited on the memory Windows had moved out).
		static uint64_t logged  = 0;
		const uint64_t  live    = KytyGcLiveBudgetValue(m_graphics);
		uint64_t        trigger = m_graphics.small_video_memory ? std::min(live, m_start_trigger_gc_memory) : live;
		// (vram-budget) Not under 4096 MiB (nor over the start-up trigger) on a small video memory heap: at 4096 a simulated
		// 8 GB card already hitched 25-97 windows against 3-17 at 5120 (GetTotalMemoryBudget), images deleted and uploaded
		// again; lower, the collector would only churn.
		if (m_graphics.small_vram_heap)
			trigger = std::max(trigger, std::min<uint64_t>(m_start_trigger_gc_memory, uint64_t {4096} << 20u));
		if ((trigger > logged ? trigger - logged : logged - trigger) >= (uint64_t {256} << 20u)) {
			logged = trigger;
			std::printf("Texture collector: collects above %llu MiB (the video memory budget now, less its headroom; in use "
			            "%llu MiB, %llu MiB left under the driver's budget)\n",
			            static_cast<unsigned long long>(trigger >> 20u), static_cast<unsigned long long>(m_total_used_memory >> 20u),
			            static_cast<unsigned long long>(m_graphics.GetVideoMemoryRoom() >> 20u));
		}
		m_trigger_gc_memory = trigger;
	}
	// A log line on where the GPU's memory goes when its use moved (checked every 60 flips): on a GPU short of video
	// memory by 256 MiB of video memory, on others by 1 GiB of video or GPU system memory (issue #14: an RX 9070 XT held
	// 16 GB of the game's GPU memory in system memory with video memory to spare, and nothing in its log said when).
	if (uint64_t video = 0, budget = 0, shared = 0; tick % 60 == 0 && m_graphics.QueryLocalVideoMemory(&video, &budget) &&
	                                                LocalPlatform::QuerySharedGpuMemory(&shared)) {
		const bool     short_of_video = m_graphics.small_video_memory;
		const uint64_t step           = short_of_video ? uint64_t {256} << 20u : uint64_t {1} << 30u;
		const auto moved = [&](uint64_t now, uint64_t logged) { return now > logged + step || now + step < logged; };
		if (moved(video, m_logged_video_memory) || (!short_of_video && moved(shared, m_logged_shared_memory))) {
			m_logged_video_memory  = video;
			m_logged_shared_memory = shared;
			char label[32];
			std::snprintf(label, sizeof(label), "flip %llu", static_cast<unsigned long long>(tick));
			m_graphics.LogVideoMemory(label);
		}
	}
	// And one on the PC's memory when the emulator's working set or private bytes moved by 1 GiB, or the memory the
	// system has left went under 2 GiB or back (checked every 60 flips; issue #47: "the emulator uses all 32 GB of my
	// RAM", with spiky frame times, and nothing in its log said how much it held).
	LogPcMemory(tick);
	// Over the budget: the images no draw or dispatch used for StaleTicks flips (~10 s), least recently used first, at
	// most MaxDeletions a flip. What the game stops drawing it stops binding: 1-1 standing, 3.2 GiB of images were used
	// in the last 2 flips, 3.0 GiB not for 160 and more, 0.05 GiB in between; turning the camera, images behind it
	// come back after a few seconds (160 flips: 5-7 MB of uploads a frame instead of 0.2-1.7, 1% low 47 -> 27 fps).
	// Further past the budget, younger images go and more a flip: a 16 GB card with a 10 GB budget climbed to 13.5 GB
	// in Boletaria (10 minutes, 1-1 to the first tower), as the images unused for 600 flips came at 40 a flip.
	// Less the room game buffers in use that got system memory need back (BufferCache::MoveSpilledBufferBack).
	constexpr uint64_t GiB           = uint64_t {1} << 30u;
	const uint64_t     normal_above  = m_trigger_gc_memory - std::min(m_trigger_gc_memory, m_buffer_cache.SpilledRoom());
	const uint64_t     collect_above = normal_above;
	// RAM / commit pressure (ram-guard.h) no longer lowers this target: images are collected only for the video memory
	// budget. Under RAM pressure the collector deleted thousands of images in use (black textures) on 16 GB PCs with
	// half the video memory free, for little RAM back.
	uint64_t           over          = m_total_used_memory > collect_above ? m_total_used_memory - collect_above : 0;
	// (vram8 gc, 11-23; KYTY_VRAM8_GC, on by default on a small video memory heap, =0 off, =1 on any card) On an 8 GB card
	// the 5120 MiB target counts the driver's own memory (~1.15 GB), the game buffers (~1.4 GB) and the free space inside
	// VMA's blocks, which deleting images does not lower: usage stayed ~1.3 GB over it with 600-860 MiB still under the
	// driver's budget, and the collector deleted images unused for 300 flips (~5 s) every flip, 1.1-1.6 GB a minute
	// (simulated 8 GB, 11-23 12:17), uploaded again when the camera came back to them. Now it collects only when the room
	// under the driver's budget plus the free space in VMA's image blocks falls under 512 MiB, by what is missing, the
	// oldest images first (600 flips; 300 when 1 GiB or more is missing).
	static const int vram8_gc_mode = [] {
		const char* v = std::getenv("KYTY_VRAM8_GC");
		return v == nullptr ? -1 : (*v == '0' ? 0 : 1);
	}();
	const bool vram8_gc = m_graphics.CanReportMemoryUsage() &&
	                      (vram8_gc_mode == 1 || (vram8_gc_mode < 0 && (m_graphics.small_vram_heap || VramTope::On())));
	// (VRAM-TOPE) Con el tope unico tambien en tarjetas de 12-16 GB: el recolector va por el tope (nivel 3) y por el hueco
	// bajo el driver, no por el disparador de antes (presupuesto menos su margen).
	uint64_t   vram8_need = 0;
	uint64_t   desg_stale = 0; // (VRAM-DESG) edad minima de lo expulsado (0: la de antes)
	bool       drv_mode   = false; // (VRAM-DRV2) expulsa solo por el techo propio: con sus limites
	static double   drv_tokens  = 0.0; // (VRAM-DRV2) bytes que aun se pueden expulsar (MaxMiBps)
	static uint64_t drv_last_ms = 0;
	// (VRAM-REPOSO, 11-25) Log 20261125-112615-361: 4665 pasadas del techo propio y solo 20 expulsadas: casi todas vacias
	// (todo usado hace < 300 fotogramas o protegido por re-subida). Tras una pasada sin candidatos, reposo de
	// VramDrv::ReposoTicks fotogramas; lo critico (hueco del driver < 128 MiB, no drv_mode) y el cambio de mundo no esperan.
	static uint64_t drv_reposo_hasta = 0;
	// (VRAM-TOPE) Cambio de mundo: durante su ventana se expulsa lo que no se ha usado desde el cambio (el mundo
	// anterior), sin el limite de MiB/s ni la espera por hueco. Lo usado despues del cambio tiene marca mayor: no se toca.
	const uint64_t mundo_tick = VramTope::On() && VramTope::g_mundo_activo.load(std::memory_order_relaxed)
	                                ? VramTope::g_mundo_tick.load(std::memory_order_relaxed)
	                                : 0;
	const bool     mundo      = mundo_tick != 0 && tick > mundo_tick + VramDrv::InUseTicks;
	if (mundo) {
		vram8_need = UINT64_MAX / 4;
		desg_stale = tick - mundo_tick; // ForEachItemBelow(tick - desg_stale) = lo no usado desde el cambio
		over       = 2 * GiB;           // MaxDeletions 160
	} else if (vram8_gc && desg_on) {
		// (VRAM-DESG) Con histeresis: empieza con menos de DesgReserve de hueco bajo el presupuesto del driver (mas el
		// libre de los bloques de imagenes de VMA) y sigue, pasada a pasada, hasta DesgReserve + DesgHyst. Solo lo que
		// lleva >= 1800 pasadas (~30 s) sin usarse; 600 (~10 s) si queda menos de 128 MiB (el driver sacaria memoria a la
		// RAM del sistema: peor que volver a subir una textura).
		static bool collecting = false;
		const uint64_t goal = collecting ? DesgReserve + DesgHyst : DesgReserve;
		if (desg_room >= goal) {
			collecting = false;
			if (m_total_used_memory >= collect_above) Vram8::gc_skipped.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		// (TOPE2) Log 20261125-074401: con el hueco del driver en ~64 MiB este camino expulsaba sin limite (80 por
		// pasada, >= 600 fotogramas) lo que se volvia a subir: 36 GB en la partida. Con el tope: solo si el nivel 3 esta
		// permitido, y siempre con los limites de VRAM_DRV2 (ver abajo).
		if (VramTope::On() && !VramTope::g_n3_permitido.load(std::memory_order_relaxed)) {
			collecting = false;
			return;
		}
		collecting = true;
		vram8_need = goal - desg_room;
		// (VRAM-8GB) Lo critico (10 s, 80 por pasada) solo con el driver a punto de sacar memoria a la RAM, no por el
		// presupuesto de [vram]: ahi solo lo de >= 30 s, sin ir y venir.
		desg_stale = desg_drv_room < (uint64_t {128} << 20u) ? 600 : 1800;
		over       = desg_drv_room < (uint64_t {128} << 20u) ? GiB : 0; // (MaxDeletions 80 en lo critico, si no 40)
		// (VRAM-DRV) Sobre el techo propio: lo que falta hasta el techo (+ histeresis), y mas joven: lo de >= 300 pasadas
		// (~5 s) sin usar, >= 120 (~2 s) si falta 512 MiB o mas. Nunca por debajo de InUseTicks: lo del fotograma actual
		// (y lo de las tablas, que se tocan cada 8) no se expulsa nunca.
		if (drv_need != 0) {
			// (VRAM-DRV2) Si solo el techo propio lo pide: limites de vram-drv.h (re-subidas protegidas, MiB/s, edad).
			const uint64_t goal_leg = DesgReserve + DesgHyst;
			const uint64_t need_leg = desg_room_leg < goal_leg ? goal_leg - desg_room_leg : 0;
			drv_mode   = VramTope::On() || (desg_drv_room >= (uint64_t {128} << 20u) && desg_room_leg >= DesgReserve); // (TOPE2)
			vram8_need = std::max(need_leg, drv_need);
			if (drv_mode && tick < drv_reposo_hasta) { // (VRAM-REPOSO)
				VramDrv::g_reposo_saltadas.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			if (drv_mode) {
				const uint64_t ms = VramReparto::NowMs();
				if (drv_last_ms != 0) {
					drv_tokens += static_cast<double>(ms - drv_last_ms) * static_cast<double>(VramDrv::MaxMiBps << 20u) / 1000.0;
				}
				drv_last_ms = ms;
				drv_tokens  = std::min(drv_tokens, static_cast<double>(VramDrv::MaxMiBps << 20u));
				if (drv_tokens < static_cast<double>(uint64_t {1} << 20u)) {
					VramDrv::g_cut_rate.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				desg_stale = drv_excess == 0 ? 1800 : drv_excess >= (uint64_t {512} << 20u) ? 120 : 300;
				if (VramTope::On()) desg_stale = std::max<uint64_t>(desg_stale, 300); // (TOPE2) nunca lo usado en ~5 s
				over       = 0; // MaxDeletions 40
			} else {
				desg_stale = std::max<uint64_t>(VramDrv::InUseTicks, drv_need >= (uint64_t {512} << 20u) ? 120 : 300);
				over       = drv_need >= (uint64_t {512} << 20u) ? 2 * GiB : GiB; // (MaxDeletions 160 / 80)
			}
			VramDrv::g_gc_runs.fetch_add(1, std::memory_order_relaxed);
			if (drv_need > VramDrv::g_gc_need_max.load(std::memory_order_relaxed)) VramDrv::g_gc_need_max.store(drv_need);
		} else if (VramTope::On()) {
			// (TOPE2) Solo el hueco del driver lo pide: tambien con los limites de VRAM_DRV2 (32 MiB/s, re-subidas
			// protegidas) y lo de >= 1800 fotogramas (600 si el driver esta a punto de paginar), 40 por pasada.
			drv_mode = true;
			if (tick < drv_reposo_hasta) { // (VRAM-REPOSO)
				VramDrv::g_reposo_saltadas.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			const uint64_t ms = VramReparto::NowMs();
			if (drv_last_ms != 0) {
				drv_tokens += static_cast<double>(ms - drv_last_ms) * static_cast<double>(VramDrv::MaxMiBps << 20u) / 1000.0;
			}
			drv_last_ms = ms;
			drv_tokens  = std::min(drv_tokens, static_cast<double>(VramDrv::MaxMiBps << 20u));
			if (drv_tokens < static_cast<double>(uint64_t {1} << 20u)) {
				VramDrv::g_cut_rate.fetch_add(1, std::memory_order_relaxed);
				return;
			}
			over = 0;
		}
		Vram8::gc_runs.fetch_add(1, std::memory_order_relaxed);
	} else if (vram8_gc) {
		if (VramTope::On() && VramTope::g_n3_bloqueado.load(std::memory_order_relaxed)) return; // (N3-BLOQ)
		constexpr uint64_t Reserve = uint64_t {512} << 20u;
		const uint64_t     slack   = m_graphics.GetVideoBlockSlack();
		const uint64_t     room    = m_graphics.GetVideoMemoryRoom() + slack;
		// (VRAM-PRES) Tambien PresMargin bajo el presupuesto de [vram] (no solo 512 MiB bajo el del driver).
		const uint64_t pres_room = pres ? (pres_budget > m_total_used_memory ? pres_budget - m_total_used_memory : 0) + slack
		                                : UINT64_MAX;
		const uint64_t need_drv  = room >= Reserve ? 0 : Reserve - room;
		const uint64_t need_pres = pres_room >= PresMargin ? 0 : PresMargin - pres_room;
		if (need_drv == 0 && need_pres == 0) {
			if (m_total_used_memory >= collect_above) Vram8::gc_skipped.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		vram8_need = std::max(need_drv, need_pres);
		over       = vram8_need >= GiB ? GiB : 0;
		Vram8::gc_runs.fetch_add(1, std::memory_order_relaxed);
	}
	const uint64_t     StaleTicks    = desg_stale != 0 ? desg_stale // (VRAM-DESG)
	                                   : over >= 2 * GiB ? 120 : over >= GiB ? 300 : 600;
	const size_t       MaxDeletions  = over >= 2 * GiB ? 160 : over >= GiB ? 80 : 40;
	// For a while after the GPU could not make an image (InsertImage), images unused for FailedRetryTicks go too: the
	// superseded ones whose memory it needs (issue #49: a 16 GB GPU full of texture-pool images).
	const bool     failing     = m_failed_tick != 0 && tick < m_failed_tick + 600;
	uint64_t       stale_ticks = failing ? std::min<uint64_t>(FailedRetryTicks, StaleTicks) : StaleTicks;
	if (VramDrv::On()) stale_ticks = std::max(stale_ticks, VramDrv::InUseTicks); // (VRAM-DRV) nunca lo de ahora
	// (N3-REAL) Con el tope, fuera del cambio de mundo: nunca lo usado en los ultimos ~5 s (300 fotogramas), en todo camino.
	// (TEX-STREAM) Cada 200 ms: nivel de limite de mips segun la VRAM del dispositivo (incluye las texturas que vma.cpp
	// manda a memoria "plain" y que NVIDIA deja en VRAM) contra el tope; linea [tex-stream] cada 5 s.
	if (TexStream::On() && m_graphics.CanReportMemoryUsage()) {
		static auto       last_ts = std::chrono::steady_clock::now();
		static const auto t0_ts   = last_ts;
		if (const auto now = std::chrono::steady_clock::now(); now - last_ts >= std::chrono::milliseconds(200)) { // (m9) 5 veces por segundo, VRAM actual
			last_ts             = now;
			const uint64_t uso  = m_graphics.GetDeviceMemoryUsage();
			const uint64_t drv  = VramDrv::g_drv_budget_bytes.load(std::memory_order_relaxed);
			uint64_t       tope = drv != 0 ? drv : m_graphics.GetTotalMemoryBudget(false);
			if (GestorMem::On() && GestorMem::Presupuesto(GestorMem::Vram) != 0) tope = GestorMem::Presupuesto(GestorMem::Vram); // (M7)
			TexStream::RegistrarEnGestor();
			TexStream::Tick(uso, tope);
			TexStream::Report(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - t0_ts).count()) + 1,
			                  m_total_used_memory, uso, tope, VramSaverGet(VramEvicted), VramSaverGet(VramEvictedBytes),
			                  VramDrv::g_reup_n.load(std::memory_order_relaxed),
			                  VramDrv::g_reup_bytes.load(std::memory_order_relaxed));
		}
	}
	if (VramTope::On() && !mundo && !failing) stale_ticks = std::max<uint64_t>(stale_ticks, 300);
	if ((!mundo && !vram8_gc && m_total_used_memory < collect_above) || tick < stale_ticks) {
		return;
	}
	// An image the collector keeps (a depth image with its stencil association; one the GPU wrote, tiled: no download
	// for its data) is passed over, not counted: never used again, kept images stay at the front of the list, and the
	// first images from the front were only those (until 10-09 no image was collected on any GPU). Deleting depth
	// deletes its stencil association too, so the traversal ends first.
	// In that while, images the GPU wrote, tiled, go too, their bytes downloaded to guest memory first (TryDownloadImage,
	// which takes no image larger than its buffer): a switch of the graphics mode leaves the old mode's targets bound by
	// nothing, and kept they left no room for the new mode's (issue #53: a 16 GB RX 7800 XT switching to the cinematic
	// mode, its images unable to go to system memory).
	const uint64_t       download_max = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download).Size() / 2;
	std::vector<ImageId> candidates;
	candidates.reserve(MaxDeletions);
	// (ram15, KYTY_NATIVE_TEXTURE_OFFLOAD; KYTY_NATIVE_OFFLOAD_EVICT_LAST=0: off) A stale image whose guest bytes were
	// dropped costs a download and its RAM back when it goes (UnregisterImage): those go after the stale ones whose bytes
	// RAM still holds (deleting those costs nothing), within the same flip's limit.
	static const bool evict_last = [] {
		const char* value = std::getenv("KYTY_NATIVE_OFFLOAD_EVICT_LAST");
		return TextureOffload::Native() && !(value != nullptr && value[0] == '0');
	}();
	const uint64_t       drv_now_ms = drv_mode ? VramReparto::NowMs() : 0; // (VRAM-DRV2)
	std::vector<ImageId> offloaded_later;
	std::vector<ImageId> native_later; // (P1) con rangos "solo en la GPU": despues de las demas
	uint64_t   prot_bytes = 0; // (N3-REAL)
	const bool proteger   = VramTope::On();
	std::vector<ImageId> rtg_ids; // (RT-GRANDE-VIEJO)
	m_lru_cache.ForEachItemBelow(tick - stale_ticks, [&](ImageId id) {
		const auto* owner = m_slot_images.try_get(id);
		if (proteger && owner != nullptr && owner->registered && GcProtegida(*owner)) { // (N3-REAL) RT/profundidad/GPU
			// (RT-GRANDE-VIEJO) Excepcion: un RT de color grande fuera de las arenas sin usar >= RtGrandeTicks sale CON su
			// descarga a la RAM del juego (si se vuelve a usar se rehace con sus bytes exactos, no con datos viejos).
			if (!RtGrandeViejo(*owner, m_lru_cache.TickOf(owner->lru_id), tick) ||
			    (owner->IsGpuModified() && (!SafeToDownload(*owner) || owner->info.data.size > RtGrandeDescargaMax))) {
				prot_bytes += owner->info.data.size;
				return false;
			}
			rtg_ids.push_back(id);
			candidates.push_back(id);
			return candidates.size() == MaxDeletions;
		}
		if (owner == nullptr || !owner->registered || owner->depth_id ||
		    (owner->IsGpuModified() && owner->info.IsTiled() && SafeToDownload(*owner) &&
		     !(failing && owner->info.data.size <= download_max))) {
			return false;
		}
		if (GcGpuSuciaRetiene(*owner)) return false; // (GC-GPU-SUCIA) pintada por la GPU y sin poder descargarse: se queda
		// (N3-BLOQ) Nativa (dispersa o con rangos solo en la GPU) usada en los ultimos ~30 s o re-subida hace < 60 s: se
		// queda en cualquier modo (antes solo el modo del techo miraba las re-subidas y la edad era 300 fotogramas, ~5 s).
		if (VramTope::On() && !mundo &&
		    (owner->IsSparse() || TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End()))) {
			if (m_lru_cache.TickOf(owner->lru_id) + VramTope::NatUsoTicks > tick) return false;
			if (VramReparto::ReupReciente(owner->info.data.address, drv_now_ms != 0 ? drv_now_ms : VramReparto::NowMs())) {
				VramDrv::g_skip_reup.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
		}
		if (drv_mode && VramReparto::ReupReciente(owner->info.data.address, drv_now_ms)) { // (VRAM-DRV2)
			VramDrv::g_skip_reup.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		// (N2+N3) Sus bytes "solo en la GPU" no estan en la RAM emulada. (P1) Va al final, como mucho 2 por pasada.
		if (TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End())) {
			if (native_later.size() < NativeEvictCap(pres ? 8u : 2u)) native_later.push_back(id); // (VRAM-PRES) 8 con presupuesto
			return false;
		}
		// (TEX-STREAM) Una sola copia: si VRAM_UNICA ya solto su RAM, la VRAM es la unica copia y no se expulsa (fuera de
		// un cambio de mundo); la VRAM de las texturas la baja el juego soltando mips (gancho 0xac5b70, tex-stream.h).
		if (TexStream::On() && !mundo && TextureOffload::Overlaps(owner->info.data.address, owner->info.data.size)) {
			TexStream::c_protegidas.fetch_add(1, std::memory_order_relaxed);
			TexStream::c_protegidas_bytes.fetch_add(owner->info.data.size, std::memory_order_relaxed);
			return false;
		}
		if (evict_last && TextureOffload::Overlaps(owner->info.data.address, owner->info.data.size)) {
			if (offloaded_later.size() < MaxDeletions) offloaded_later.push_back(id);
			return false;
		}
		candidates.push_back(id);
		return candidates.size() == MaxDeletions;
	});
	if (!offloaded_later.empty()) {
		std::lock_guard offload_lock(TextureOffload::Get().mutex);
		TextureOffload::Get().evict_deferred += offloaded_later.size();
	}
	for (const auto id: offloaded_later) {
		if (candidates.size() >= MaxDeletions) break;
		candidates.push_back(id);
	}
	for (const auto id: native_later) candidates.push_back(id); // (P1) fuera del limite MaxDeletions (max 2)
	if (proteger) VramDrv::g_prot_bytes.store(prot_bytes, std::memory_order_relaxed); // (N3-REAL)
	if (drv_mode && !mundo && candidates.empty()) { // (VRAM-REPOSO) pasada vacia: descansa ReposoTicks fotogramas
		drv_reposo_hasta = tick + VramDrv::ReposoTicks;
		VramDrv::g_reposo_vacias.fetch_add(1, std::memory_order_relaxed);
	}
	uint64_t vram8_freed = 0;
	// (GC-TOPE, KYTY_GC_TOPE_MS, por defecto 4; 0: sin tope) Log 20261125-011726: los fotogramas de 76.8 y 104.8 ms
	// (cpu 74-98, queue_run 16-19, sin gpu_wait) son esta pasada fuera de queue_run: las expulsiones del [vram] saltan
	// justo ahi (49 -> 227, 377 -> 448 imagenes; 184 MiB) y cada imagen quitada deshace la vigilancia de sus paginas
	// (UnregisterImage, coste por pagina). Borrar es opcional (lo que queda sigue viejo y sale en la pasada siguiente):
	// pasado el tope, el resto espera al proximo fotograma. La VRAM se libera igual, repartida en varios fotogramas.
	static const int64_t gc_tope_us = [] {
		const char* v  = std::getenv("KYTY_GC_TOPE_MS");
		const long  ms = v != nullptr && *v != 0 ? std::atol(v) : 0L; // (04:20) sin tope por defecto: con 4 ms la VRAM quedaba sobre el presupuesto
		std::printf("[gc-tope] recolector de texturas: %s (KYTY_GC_TOPE_MS=%ld; 0: sin tope)\n", ms > 0 ? "tope por fotograma" : "sin tope", ms);
		std::fflush(stdout);
		return ms > 0 ? static_cast<int64_t>(ms) * 1000 : int64_t {0};
	}();
	static uint64_t gc_tope_cortes = 0, gc_tope_dejadas = 0;
	const auto      gc_tope_inicio = std::chrono::steady_clock::now();
	size_t          gc_tope_hechas = 0;
	for (const auto id: candidates) {
		if (gc_tope_us > 0 && gc_tope_hechas != 0 &&
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - gc_tope_inicio).count() >=
		        gc_tope_us) {
			gc_tope_dejadas += candidates.size() - gc_tope_hechas;
			if ((++gc_tope_cortes & 63u) == 1u) {
				std::printf("[gc-tope] pasada cortada a los %lld ms: %zu borradas, %zu para el siguiente fotograma (cortes %llu, "
				            "aplazadas en total %llu)\n",
				            static_cast<long long>(gc_tope_us / 1000), gc_tope_hechas, candidates.size() - gc_tope_hechas,
				            static_cast<unsigned long long>(gc_tope_cortes), static_cast<unsigned long long>(gc_tope_dejadas));
				std::fflush(stdout);
			}
			break;
		}
		++gc_tope_hechas;
		auto owner = m_slot_images.try_get(id);
		if (!mundo && (vram8_gc ? vram8_freed >= vram8_need : m_total_used_memory < collect_above)) {
			break;
		}
		if (drv_mode && static_cast<double>(vram8_freed) >= drv_tokens) { // (VRAM-DRV2) MaxMiBps
			VramDrv::g_cut_rate.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		if (owner == nullptr || !owner->registered) {
			continue;
		}
		const bool ev_nativa = owner->IsSparse() || TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End()); // (N3-REAL)
		// (P1) Recuperar antes sus bytes "solo en la GPU" a la RAM; si no se puede, se queda. FlushAndWait: releer.
		if (TexNativa::On() && TexNativa::AnyOwned(owner->info.data.address, owner->info.data.End())) {
			if (!NativeRestoreBeforeDelete(*owner, true)) continue;
			owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered) continue;
		}
		if (owner->IsGpuModified()) {
			const bool rtg = !rtg_ids.empty() && std::find(rtg_ids.begin(), rtg_ids.end(), id) != rtg_ids.end(); // (W14)
			g_rtg_descarga_propia = rtg;
			const bool bajada     = !SafeToDownload(*owner) || TryDownloadImage(id);
			g_rtg_descarga_propia = false;
			if (!bajada) {
				continue;
			}
			owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered) continue;
			if (!SafeToDownload(*owner)) GcGpuSuciaPerdida(*owner); // (GC-GPU-SUCIA) borrada sin descarga
	RtNativo::Informe(); // (RT-NATIVO) linea [rt-nativo] cada 30 s
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
		if (!rtg_ids.empty() && std::find(rtg_ids.begin(), rtg_ids.end(), id) != rtg_ids.end()) { // (RT-GRANDE-VIEJO)
			static uint64_t rtg_n = 0, rtg_bytes = 0;
			rtg_n++;
			rtg_bytes += owner->info.data.size;
			if (rtg_n <= 32) {
				std::printf("[rt-grande-viejo] liberado 0x%llx %ux%u f%u %llu MiB, sin usar %llu fotogramas (total %llu, %llu MiB)\n",
				            static_cast<unsigned long long>(owner->info.data.address), owner->info.extent.width,
				            owner->info.extent.height, static_cast<uint32_t>(owner->info.pixel_format),
				            static_cast<unsigned long long>(owner->info.data.size >> 20u),
				            static_cast<unsigned long long>(tick - std::min(tick, m_lru_cache.TickOf(owner->lru_id))),
				            static_cast<unsigned long long>(rtg_n), static_cast<unsigned long long>(rtg_bytes >> 20u));
				std::fflush(stdout);
			}
		}
		VramReparto::Evicted(owner->info.data.address, owner->info.data.size); // (VRAM-REPARTO)
		{ // (N3-REAL) que tipo se expulsa
			const bool offl = evict_last && TextureOffload::Overlaps(owner->info.data.address, owner->info.data.size);
			(ev_nativa ? VramDrv::g_ev_nat : offl ? VramDrv::g_ev_offl : VramDrv::g_ev_tex)
			    .fetch_add((owner->IsSparse() && SparseIsPerLevel(owner->backing)) ? SparseBoundBytes(owner->backing) : owner->info.data.size, std::memory_order_relaxed); // (SPARSE-POOL) nativa: lo ligado real
		}
		VramSaverAdd(VramEvicted, 1);
		VramSaverAdd(VramEvictedBytes, (owner->IsSparse() && SparseIsPerLevel(owner->backing)) ? SparseBoundBytes(owner->backing) : owner->info.data.size); // (SPARSE-POOL)
		// (VRAM-PRES) Una nativa libera lo ligado a sus mips (no su tamano en la RAM del juego).
		if (pres && owner->IsSparse() && SparseIsPerLevel(owner->backing)) {
			const uint64_t real = SparseBoundBytes(owner->backing);
			vram8_freed += real;
			m_pres_native_evicted++;
			m_pres_native_evicted_bytes += real;
		} else {
			vram8_freed += owner->info.data.size;
		}
		if (evict_last && TextureOffload::Overlaps(owner->info.data.address, owner->info.data.size)) {
			std::lock_guard offload_lock(TextureOffload::Get().mutex);
			TextureOffload::Get().evict_offloaded++;
		}
		if (mundo) { // (VRAM-TOPE)
			VramTope::g_mundo_n.fetch_add(1, std::memory_order_relaxed);
			VramTope::g_mundo_bytes.fetch_add(owner->info.data.size, std::memory_order_relaxed);
		}
		DeleteImage(id);
		LiveCounters::Add(LiveCounters::GcImageDeletes);
	}
	if (drv_mode) drv_tokens = std::max(0.0, drv_tokens - static_cast<double>(vram8_freed)); // (VRAM-DRV2)
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
