#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/image/imageDump.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/nanScanner.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/stallWatch.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;
// Overlapping aliases (transient render targets sharing memory) stay registered until unused for
// this many presented frames as well; see AliasAgeByFrames.
constexpr uint64_t AliasFramesBeforeRemoval = 4;

// An overlapped alias was freed once unused for NumFramesBeforeRemoval scheduler ticks. A title
// submits hundreds of times a frame, so that is usually within the frame that used it: the
// aliases of a transient image pool (Wolverine: ~4 a frame) were destroyed and recreated from
// guest memory every frame. They now also have to be unused for AliasFramesBeforeRemoval
// presented frames, and live aliases follow a single-owner rule (CommitGpuWrite).
// KYTY_IMAGE_ALIAS_AGE=0 (or =ticks) restores the tick rule alone.
[[nodiscard]] bool AliasAgeByFrames() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_IMAGE_ALIAS_AGE");
		return value == nullptr ||
		       (std::strcmp(value, "0") != 0 && std::strcmp(value, "ticks") != 0);
	}();
	return enabled;
}

// The collector writes a GPU-written image back to guest memory before it frees it (under
// memory pressure). Whether the image is CPU-dirty is checked when the copy is recorded, but the
// bytes land only once the GPU is done with that tick, and the image is untracked by then: a
// guest that reused the memory in between (a retired render target becoming a CPU-written vertex
// table, as in Jetsku's 919e712a) had it overwritten with stale pixels. The guest bytes are hashed
// when the copy is recorded and again before it is published; if they changed, the write-back
// is dropped. KYTY_EVICT_WRITEBACK_GUARD=0 publishes unconditionally again.
[[nodiscard]] bool EvictWriteBackGuard() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_EVICT_WRITEBACK_GUARD");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Hash of the guest backing bytes of a range, read in chunks; false if a chunk is unreadable.
[[nodiscard]] bool HashGuestBacking(uint64_t address, uint64_t size, uint64_t& hash) {
	constexpr uint64_t   Chunk = 64 * 1024;
	std::vector<uint8_t> bytes(static_cast<size_t>(std::min(size, Chunk)));
	uint64_t             seed = 0;
	for (uint64_t done = 0; done < size;) {
		const auto part = std::min(size - done, Chunk);
		if (!LibKernel::Memory::TryReadBacking(address + done, bytes.data(), part)) {
			return false;
		}
		seed = XXH3_64bits_withSeed(bytes.data(), static_cast<size_t>(part), seed);
		done += part;
	}
	hash = seed;
	return true;
}

// HashGuestBacking's hash of bytes in host memory.
[[nodiscard]] uint64_t HashBytes(const uint8_t* bytes, uint64_t size) {
	constexpr uint64_t Chunk = 64 * 1024;
	uint64_t           seed  = 0;
	for (uint64_t done = 0; done < size;) {
		const auto part = std::min(size - done, Chunk);
		seed            = XXH3_64bits_withSeed(bytes + done, static_cast<size_t>(part), seed);
		done += part;
	}
	return seed;
}

// The hash of the bytes the last read-back for the CPU wrote at a guest address (priority
// thread): a read-back recorded before that one landed sees these bytes, not a guest write.
// Entries go when the memory is unmapped.
struct LandedReadbacks {
	std::mutex                   mutex;
	std::map<uint64_t, uint64_t> hashes;
};

LandedReadbacks& GetLandedReadbacks() {
	// Never destroyed: read-backs may still land during static destruction.
	static auto* landed = new LandedReadbacks;
	return *landed;
}

[[nodiscard]] uint64_t LandedReadbackHash(uint64_t address) {
	auto&           landed = GetLandedReadbacks();
	std::lock_guard lock(landed.mutex);
	const auto      found = landed.hashes.find(address);
	return found != landed.hashes.end() ? found->second : 0;
}

void NoteLandedReadback(uint64_t address, const uint8_t* bytes, uint64_t size) {
	const auto      hash   = HashBytes(bytes, size);
	auto&           landed = GetLandedReadbacks();
	std::lock_guard lock(landed.mutex);
	landed.hashes[address] = hash;
}

void ForgetLandedReadbacks(uint64_t address, uint64_t size) {
	auto&           landed = GetLandedReadbacks();
	std::lock_guard lock(landed.mutex);
	landed.hashes.erase(landed.hashes.lower_bound(address),
	                    landed.hashes.lower_bound(address + size));
}

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.info.IsDepth()
	                           ? TextureGetSurfaceFormatInfo(desc.info.guest_format).vk_format
	                           : desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		if (desc.type != TextureCache::BindingType::RenderTarget || !metadata.clear_register_valid) {
			return false;
		}
		// The clear word holds texel bits of the guest's layout (a 10_11_11 target's differ from
		// the host format's).
		const auto clear_word =
		    GuestColorWordToHost(metadata.clear_word, desc.info.GetColorTransform());
		if (DecodePackedColorClear(format, clear_word, clear)) {
			return true;
		}
		// KYTY_CLEAR_REGISTER_WIDE: 64-bit, 16-bit-channel and B10G11R11 targets (both words).
		if (!DecodePackedColorClear64(format, clear_word, metadata.clear_word1, clear)) {
			return false;
		}
		static std::mutex                   logged_mutex;
		static std::unordered_set<uint32_t> logged;
		if (std::lock_guard lock(logged_mutex); logged.insert(static_cast<uint32_t>(format)).second) {
			LOGF("Register clear of format %u decoded with both clear words (was dropped)\n",
			     static_cast<uint32_t>(format));
		}
		return true;
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

[[nodiscard]] float HalfBitsToFloat(uint32_t bits) {
	const uint32_t sign     = (bits & 0x8000u) << 16u;
	const uint32_t exponent = (bits >> 10u) & 0x1fu;
	uint32_t       mantissa = bits & 0x3ffu;
	if (exponent == 0x1fu) {
		return std::bit_cast<float>(sign | 0x7f800000u | (mantissa << 13u));
	}
	if (exponent != 0) {
		return std::bit_cast<float>(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
	}
	if (mantissa == 0) {
		return std::bit_cast<float>(sign);
	}
	uint32_t shift = 0;
	while ((mantissa & 0x400u) == 0) {
		mantissa <<= 1u;
		++shift;
	}
	return std::bit_cast<float>(sign | ((113u - shift) << 23u) | ((mantissa & 0x3ffu) << 13u));
}

// Whether a clear to these float bits keeps them: zero or a normal number. A clear need not keep
// a NaN's payload, an infinity or a denormal (the dispatch's store would).
[[nodiscard]] bool ExactFloatBits(uint32_t bits) {
	const auto exponent = (bits >> 23u) & 0xffu;
	return bits == 0 || (exponent != 0 && exponent != 0xffu);
}

[[nodiscard]] bool ExactHalfBits(uint32_t bits) {
	const auto exponent = (bits >> 10u) & 0x1fu;
	return bits == 0 || (exponent != 0 && exponent != 0x1fu);
}

// A dword fill of a color image's memory as a clear of it: the formats DecodeColorDwordFill
// knows, and for the others a fill of zero (every texel's bits zero), and the 8-, 16-bit-channel
// and two- and four-channel 32-bit formats whose texels the word repeats (a byte or half-word
// texel the word holds over and over, or a texel of whole words). Float channels must be zero or
// normal numbers. False for the rest (the fill then runs or uploads as before).
[[nodiscard]] bool DecodeFillWord(vk::Format format, uint32_t packed, vk::ClearColorValue& clear) {
	if (format == vk::Format::eR32Sfloat && !ExactFloatBits(packed)) {
		return false;
	}
	if (DecodeColorDwordFill(format, packed, clear)) {
		return true;
	}
	const auto unorm = [](uint32_t value, uint32_t bits) {
		return static_cast<float>(value) / static_cast<float>((1u << bits) - 1u);
	};
	const uint32_t      low         = packed & 0xffffu;
	const uint32_t      high        = packed >> 16u;
	const bool          same_bytes  = packed == (packed & 0xffu) * 0x01010101u;
	const bool          same_halves = low == high;
	vk::ClearColorValue next {};
	switch (format) {
		case vk::Format::eR8Unorm:
			if (!same_bytes) return false;
			next.float32[0] = unorm(packed & 0xffu, 8);
			break;
		case vk::Format::eR8Uint:
			if (!same_bytes) return false;
			next.uint32[0] = packed & 0xffu;
			break;
		case vk::Format::eR8G8Unorm:
			if (!same_halves) return false;
			next.float32[0] = unorm(low & 0xffu, 8);
			next.float32[1] = unorm(low >> 8u, 8);
			break;
		case vk::Format::eR8G8Uint:
			if (!same_halves) return false;
			next.uint32[0] = low & 0xffu;
			next.uint32[1] = low >> 8u;
			break;
		case vk::Format::eR16Unorm:
			if (!same_halves) return false;
			next.float32[0] = unorm(low, 16);
			break;
		case vk::Format::eR16Uint:
			if (!same_halves) return false;
			next.uint32[0] = low;
			break;
		case vk::Format::eR16Sfloat:
			if (!same_halves || !ExactHalfBits(low)) return false;
			next.float32[0] = HalfBitsToFloat(low);
			break;
		case vk::Format::eR16G16Unorm:
			next.float32[0] = unorm(low, 16);
			next.float32[1] = unorm(high, 16);
			break;
		case vk::Format::eR16G16Uint:
			next.uint32[0] = low;
			next.uint32[1] = high;
			break;
		case vk::Format::eR16G16Sfloat:
			if (!ExactHalfBits(low) || !ExactHalfBits(high)) return false;
			next.float32[0] = HalfBitsToFloat(low);
			next.float32[1] = HalfBitsToFloat(high);
			break;
		case vk::Format::eR8G8B8A8Uint:
			for (uint32_t channel = 0; channel < 4; ++channel) {
				next.uint32[channel] = (packed >> (channel * 8u)) & 0xffu;
			}
			break;
		case vk::Format::eR16G16B16A16Unorm:
			next.float32[0] = next.float32[2] = unorm(low, 16);
			next.float32[1] = next.float32[3] = unorm(high, 16);
			break;
		case vk::Format::eR16G16B16A16Uint:
			next.uint32[0] = next.uint32[2] = low;
			next.uint32[1] = next.uint32[3] = high;
			break;
		case vk::Format::eR16G16B16A16Sfloat:
			if (!ExactHalfBits(low) || !ExactHalfBits(high)) return false;
			next.float32[0] = next.float32[2] = HalfBitsToFloat(low);
			next.float32[1] = next.float32[3] = HalfBitsToFloat(high);
			break;
		case vk::Format::eR32G32Uint:
		case vk::Format::eR32G32B32A32Uint: next.uint32.fill(packed); break;
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
			if (!ExactFloatBits(packed)) return false;
			next.float32.fill(std::bit_cast<float>(packed));
			break;
		default:
			// Zero bits are zero in any uncompressed color format.
			if (packed != 0 || !vk::isColor(format) || vk::isCompressed(format) ||
			    vk::planeCount(format) > 1 || vk::texelsPerBlock(format) != 1) {
				return false;
			}
			break;
	}
	clear = next;
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

// Round 123: `info` (the requested image with the layers of the cached one it replaces) gets a
// range that holds those layers: the requested slices, stacked from the same address (as a depth
// target's are), and the stencil and HTile ranges likewise. False when the requested image is not
// such a stack (another first address, mips, a range its layers do not divide), and the caller
// keeps the requested layers.
[[nodiscard]] bool GrowDepthLayers(ImageInfo& info, const ImageInfo& requested,
                                   const ImageInfo& cached) {
	const auto old_layers = requested.resources.layers;
	const auto new_layers = info.resources.layers;
	if (info.resources.levels != requested.resources.levels || requested.resources.levels != 1 ||
	    old_layers == 0 || new_layers <= old_layers || info.IsVolume() ||
	    requested.data.address != cached.data.address || requested.data.size % old_layers != 0 ||
	    requested.mip_layout[0].offset != 0 ||
	    requested.mip_layout[0].size != requested.data.size ||
	    requested.stencil.size % old_layers != 0 ||
	    requested.metadata.range.size % old_layers != 0) {
		return false;
	}
	const auto grow = [&](GuestRange range) -> std::optional<GuestRange> {
		if (range.Empty()) {
			return range;
		}
		const auto slice = range.size / old_layers;
		if (slice > UINT64_MAX / new_layers) {
			return std::nullopt;
		}
		const GuestRange grown {range.address, slice * new_layers};
		return grown.Valid() ? std::optional {grown} : std::nullopt;
	};
	const auto data     = grow(requested.data);
	const auto stencil  = grow(requested.stencil);
	const auto metadata = grow(requested.metadata.range);
	if (!data || !stencil || !metadata) {
		return false;
	}
	info.data               = *data;
	info.stencil            = *stencil;
	info.metadata.range     = *metadata;
	info.mip_layout[0].size = data->size;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		LOGF("TextureCache: depth image 0x%012" PRIx64 " %ux%u of %u layers replaced for a "
		     "binding of %u: the new one keeps %u layers, 0x%" PRIx64 " bytes (round 123)\n",
		     requested.data.address, requested.extent.width, requested.extent.height,
		     cached.resources.layers, old_layers, new_layers, data->size);
	}
	return true;
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

// Where the float bit fields sit in a word of a host format NanScanner reads; none for the
// others (block-compressed and integer formats, and the floats that cannot hold infinity or NaN).
[[nodiscard]] std::optional<NanScanner::Packing> NonFinitePacking(vk::Format format) {
	switch (format) {
		case vk::Format::eB10G11R11UfloatPack32: return NanScanner::Packing::B10G11R11;
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat: return NanScanner::Packing::TwoHalves;
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat: return NanScanner::Packing::Float32;
		default: return std::nullopt;
	}
}

// KYTY_GPU_SCAN_MB (default 512): the most a GPU scan copies of one image (every mip).
[[nodiscard]] uint64_t GpuScanLimit() {
	static const uint64_t limit = [] {
		const char* value = std::getenv("KYTY_GPU_SCAN_MB");
		return (value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : 512ull)
		       << 20u;
	}();
	return limit;
}

// How a GPU scan copies an image: each mip, every layer, tightly packed one after the other, a
// mip starting at a multiple of the texel size (as a copy's buffer offset must). `bytes` is 0 when
// the scan does not apply.
struct NonFiniteLayout {
	std::vector<vk::BufferImageCopy> regions;
	std::vector<uint32_t>            words; // a layer's words, for each mip
	uint64_t                         bytes   = 0;
	NanScanner::Packing              packing = NanScanner::Packing::Float32;
};

[[nodiscard]] NonFiniteLayout NonFiniteScanLayout(const Image& image, uint64_t max_counters) {
	NonFiniteLayout layout;
	const auto&     backing = image.backing;
	const auto      packing = NonFinitePacking(backing.format);
	if (!packing || image.depth_id || backing.image == nullptr || backing.samples != 1 ||
	    backing.image_type != vk::ImageType::e2D || image.info.IsDepth() ||
	    !(backing.usage & vk::ImageUsageFlagBits::eTransferSrc)) {
		return layout;
	}
	const uint32_t layers      = std::max(backing.layers, 1u);
	const uint32_t mips        = std::max(backing.mip_levels, 1u);
	const uint32_t texel_bytes = vk::blockSize(backing.format);
	if (texel_bytes == 0 || texel_bytes % sizeof(uint32_t) != 0 ||
	    uint64_t {layers} * mips > max_counters) {
		return layout;
	}
	const uint64_t alignment = uint64_t {texel_bytes} * 64u;
	uint64_t       total     = 0;
	layout.regions.resize(mips);
	layout.words.resize(mips);
	for (uint32_t mip = 0; mip < mips; mip++) {
		const uint32_t width  = std::max(backing.extent.width >> mip, 1u);
		const uint32_t height = std::max(backing.extent.height >> mip, 1u);
		const uint64_t words  = uint64_t {width} * height * (texel_bytes / sizeof(uint32_t));
		if (words * layers > UINT32_MAX) {
			return {};
		}
		total                   = (total + alignment - 1u) / alignment * alignment;
		layout.words[mip]       = static_cast<uint32_t>(words);
		auto& region            = layout.regions[mip];
		region.bufferOffset     = total;
		region.imageSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, layers};
		region.imageExtent      = vk::Extent3D {width, height, 1u};
		total += words * sizeof(uint32_t) * layers;
	}
	layout.bytes   = total;
	layout.packing = *packing;
	return layout;
}

// The runs of set entries: " <first>-<last>" each (" <first>" for one), up to `most` runs.
[[nodiscard]] std::string SetRuns(const std::vector<uint8_t>& set, size_t most) {
	std::string text;
	size_t      runs = 0;
	for (size_t first = 0; first < set.size();) {
		if (set[first] == 0) {
			first++;
			continue;
		}
		size_t last = first;
		while (last + 1u < set.size() && set[last + 1u] != 0) {
			last++;
		}
		if (runs++ < most) {
			text += last == first ? fmt::format(" {}", first) : fmt::format(" {}-{}", first, last);
		}
		first = last + 1u;
	}
	if (runs > most) {
		text += fmt::format(" (and {} more runs)", runs - most);
	}
	return text;
}

// The layers of a scanned image that hold words that are not finite in any of its mips.
struct NonFiniteLayers {
	std::vector<uint8_t>  layers; // a flag a layer
	std::vector<uint32_t> by_mip; // how many layers of each mip
	uint32_t              count = 0;
	uint32_t              cubes = 0; // of the arrays of whole cubes: the cubes with such a layer
	uint64_t              words = 0;
};

[[nodiscard]] NonFiniteLayers FindNonFiniteLayers(const TextureCache::NonFiniteCounts& counts) {
	NonFiniteLayers found;
	found.layers.assign(counts.layers, 0);
	found.by_mip.assign(counts.mips, 0);
	for (uint32_t mip = 0; mip < counts.mips; mip++) {
		for (uint32_t layer = 0; layer < counts.layers; layer++) {
			const auto count = counts.Count(mip, layer);
			if (count != 0) {
				found.layers[layer] = 1;
				found.by_mip[mip]++;
				found.words += count;
			}
		}
	}
	for (uint32_t layer = 0; layer < counts.layers; layer++) {
		found.count += found.layers[layer];
	}
	if (counts.layers % 6u == 0) {
		for (uint32_t cube = 0; cube < counts.layers / 6u; cube++) {
			bool any = false;
			for (uint32_t face = 0; face < 6u; face++) {
				any = any || found.layers[cube * 6u + face] != 0;
			}
			found.cubes += any ? 1u : 0u;
		}
	}
	return found;
}

// "<n> of <layers> layers (<c> of <cubes> cubes) hold texels that are not finite (by mip: ...)"
[[nodiscard]] std::string DescribeNonFinite(const TextureCache::NonFiniteCounts& counts,
                                            const NonFiniteLayers&               found) {
	auto text = fmt::format("{} of {} layers", found.count, counts.layers);
	if (counts.layers % 6u == 0) {
		text += fmt::format(" ({} of {} cubes)", found.cubes, counts.layers / 6u);
	}
	text += " hold texels that are not finite";
	if (found.count != 0) {
		text += fmt::format(", {} words in all; layers by mip:", found.words);
		for (const auto layers: found.by_mip) {
			text += fmt::format(" {}", layers);
		}
	}
	return text;
}

[[nodiscard]] std::string DescribeScanned(const TextureCache::NonFiniteCounts& counts) {
	return fmt::format("0x{:012x} {}x{}x{} mips {} fmt {} (guest {})", counts.address, counts.width,
	                   counts.height, counts.layers, counts.mips, counts.format,
	                   counts.guest_format);
}

// What the NaN check last found in each image, by guest address and shape (priority thread).
struct NanCheckResults {
	std::mutex                                                                         mutex;
	std::map<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>, std::vector<uint8_t>> layers;
};

// Logged at an image's first check, and when the layers that hold texels that are not finite
// change.
void ReportNanCheck(NanCheckResults& results, uint64_t frame,
                    const TextureCache::NonFiniteCounts& counts) {
	const auto       found = FindNonFiniteLayers(counts);
	const auto       key = std::tuple {counts.address, counts.width, counts.height, counts.layers};
	std::scoped_lock lock {results.mutex};
	const auto [entry, first] = results.layers.try_emplace(key);
	auto& before              = entry->second;
	if (!first && before == found.layers) {
		return;
	}
	std::string change;
	if (!first) {
		std::vector<uint8_t> became(counts.layers, 0);
		std::vector<uint8_t> cleared(counts.layers, 0);
		for (uint32_t layer = 0; layer < counts.layers; layer++) {
			became[layer]  = found.layers[layer] != 0 && before[layer] == 0 ? 1u : 0u;
			cleared[layer] = found.layers[layer] == 0 && before[layer] != 0 ? 1u : 0u;
		}
		const auto became_runs  = SetRuns(became, 24);
		const auto cleared_runs = SetRuns(cleared, 24);
		change                  = fmt::format("; since the last check{}{}{}{}",
                             became_runs.empty() ? "" : ", not finite now:", became_runs,
                             cleared_runs.empty() ? "" : ", finite again:", cleared_runs);
	} else if (found.count != 0) {
		change = fmt::format(": layers{}", SetRuns(found.layers, 24));
	}
	LOGF("TextureCache: NaN check frame %" PRIu64 " %s: %s%s\n", frame,
	     DescribeScanned(counts).c_str(), DescribeNonFinite(counts, found).c_str(), change.c_str());
	before = found.layers;
}

} // namespace

// The GPU scan (TextureCache::ScanNonFiniteGpu): the NaN-scan pipeline, a scratch buffer each mip
// of an image is copied into tightly packed, the counters, and the ring the counters come back
// through. The scratch grows to the largest image scanned so far, up to KYTY_GPU_SCAN_MB (default
// 512), and is freed after IdleFrames frames without a scan. The callbacks keep what they read
// from alive.
struct TextureCache::GpuScan {
	static constexpr uint64_t RingBytes   = 16ull << 20u;
	static constexpr uint32_t MaxCounters = 1u << 18u;
	static constexpr uint64_t IdleFrames  = 600;

	GpuScan(GraphicContext& graphics, CommandScheduler& scheduler)
	    : scanner(graphics), ring(std::make_shared<StreamBuffer>(
	                             graphics, scheduler, MemoryUsage::Download, RingBytes)) {}

	NanScanner                    scanner;
	std::shared_ptr<StreamBuffer> ring;
	std::shared_ptr<Buffer>       scratch;
	std::shared_ptr<Buffer>       counts;
	uint64_t                      last_frame = 0; // presented frame of the last scan
};

// KYTY_NAN_CHECK (default on; 0 or off turns it off): see TextureCache::TraceFrameEnd. Candidates
// are the images CommitGpuWrite saw (GPU thread, under the cache's lock). An image over
// KYTY_NAN_CHECK_MB (default 256; 64 before round 110, which left out the 128x128 light probe
// arrays of about 127 MB that a race's smoke is lit from) in all its mips is not checked (said
// once).
struct TextureCache::NanCheck {
	static constexpr uint64_t Interval = 30; // presented frames between two checks of an image

	struct Candidate {
		uint64_t address  = 0;
		uint64_t frame    = 0; // of the last check
		uint64_t content  = 0; // the image's content serial then
		bool     checked  = false;
		bool     rejected = false; // it cannot be checked
	};
	std::unordered_map<ImageId, Candidate> candidates;
	std::shared_ptr<NanCheckResults>       results   = std::make_shared<NanCheckResults>();
	uint64_t                               max_bytes = 256ull << 20u;

	[[nodiscard]] static std::unique_ptr<NanCheck> FromEnvironment() {
		std::string value =
		    std::getenv("KYTY_NAN_CHECK") != nullptr ? std::getenv("KYTY_NAN_CHECK") : "";
		std::transform(value.begin(), value.end(), value.begin(),
		               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		if (value == "0" || value == "off" || value == "false" || value == "no") {
			LOGF("TextureCache: the NaN check is off (KYTY_NAN_CHECK=%s)\n", value.c_str());
			return nullptr;
		}
		auto check = std::make_unique<NanCheck>();
		if (const char* mb = std::getenv("KYTY_NAN_CHECK_MB"); mb != nullptr && *mb != '\0') {
			check->max_bytes = std::strtoull(mb, nullptr, 10) << 20u;
		}
		return check;
	}
};

// The last snapshot of an image KYTY_TRACE_IMAGES follows (a glyph a layer), by guest address;
// written as snapshots land (the scheduler's completion work).
struct TextureCache::TraceHistory {
	std::mutex  mutex;
	std::string glyphs;
};

// KYTY_TRACE_IMAGES=<item>[,...], an item <width>x<height>[x<layers>] (the images of that shape)
// or 0x<address>[+<size>] (the images over any byte of that range): these images are followed
// through the cache from their creation on: uploads (with what their guest bytes hold), CPU
// writes, GPU buffer writes over them (at their base or inside them), their bindings as render or
// depth targets and the viewports and scissors of the draws into them, copies, write-backs and
// frees ("Trace image #<n> ..."); and at the end of each guest frame in which one changed, a
// snapshot of its mip 0 as the GPU holds it: the layers that changed and the shaders that wrote it
// (TraceFrameEnd), and a picture of it when KYTY_DUMP_IMAGES matches it too (up to
// KYTY_DUMP_IMAGES_TRACE_MAX, default 24, image-dumps/trace_*.png). Up to
// KYTY_TRACE_IMAGES_SNAPSHOTS (default 600) snapshots in all.
struct TextureCache::TraceState {
	struct Shape {
		uint32_t width  = 0;
		uint32_t height = 0;
		uint32_t layers = 0; // 0: any
	};
	struct Range {
		uint64_t address = 0;
		uint64_t size    = 1;
	};
	struct Followed {
		uint32_t                        serial           = 0;
		bool                            changed          = false;
		bool                            unscanned_logged = false;
		uint32_t                        cpu_writes       = 0;
		uint32_t                        guest_logs       = 0;
		uint32_t                        inner_writes     = 0;
		uint64_t                        bind_frame       = UINT64_MAX; // last target binding logged
		uint64_t                        draw_frame       = UINT64_MAX; // draw_states' frame
		std::set<std::string>           draw_states;
		uint32_t                        target_logs = 0; // bindings and draws logged
		std::map<std::string, uint32_t> writers; // since the last snapshot
	};
	std::vector<Shape>                                              shapes;
	std::vector<Range>                                              ranges;
	std::unordered_map<ImageId, Followed>                           images;
	std::map<uint64_t, std::shared_ptr<TextureCache::TraceHistory>> history;
	std::map<uint64_t, std::string> guest_glyphs; // the guest bytes last logged, by address
	uint32_t                        serials       = 0;
	uint32_t                        snapshots     = 0;
	uint32_t                        max_snapshots = 600;

	[[nodiscard]] bool Matches(const ImageInfo& info) const {
		return std::any_of(shapes.begin(), shapes.end(),
		                   [&](const Shape& shape) {
			                   return info.extent.width == shape.width &&
			                          info.extent.height == shape.height &&
			                          (shape.layers == 0 || info.resources.layers == shape.layers);
		                   }) ||
		       std::any_of(ranges.begin(), ranges.end(), [&](const Range& range) {
			       return info.data.size != 0 && info.data.address < range.address + range.size &&
			              range.address < info.data.address + info.data.size;
		       });
	}

	[[nodiscard]] static std::unique_ptr<TraceState> FromEnvironment() {
		const char* value = std::getenv("KYTY_TRACE_IMAGES");
		if (value == nullptr || *value == '\0') {
			return nullptr;
		}
		auto        state = std::make_unique<TraceState>();
		std::string text  = value;
		for (size_t begin = 0; begin < text.size();) {
			const auto end   = std::min(text.find(',', begin), text.size());
			const auto token = text.substr(begin, end - begin);
			begin            = end + 1u;
			if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) {
				Range range;
				char* rest    = nullptr;
				range.address = std::strtoull(token.c_str() + 2, &rest, 16);
				if (rest != nullptr && *rest == '+') {
					range.size = std::max<uint64_t>(std::strtoull(rest + 1, nullptr, 16), 1u);
				}
				state->ranges.push_back(range);
				continue;
			}
			Shape       shape;
			const char* cursor = token.c_str();
			char*       next   = nullptr;
			shape.width        = static_cast<uint32_t>(std::strtoul(cursor, &next, 10));
			if (next == cursor || *next != 'x') {
				continue;
			}
			cursor       = next + 1;
			shape.height = static_cast<uint32_t>(std::strtoul(cursor, &next, 10));
			if (next == cursor) {
				continue;
			}
			if (*next == 'x') {
				shape.layers = static_cast<uint32_t>(std::strtoul(next + 1, nullptr, 10));
			}
			if (shape.width != 0 && shape.height != 0) {
				state->shapes.push_back(shape);
			}
		}
		if (const char* snapshots = std::getenv("KYTY_TRACE_IMAGES_SNAPSHOTS")) {
			state->max_snapshots = static_cast<uint32_t>(std::strtoul(snapshots, nullptr, 10));
		}
		std::string shapes;
		for (const auto& shape: state->shapes) {
			shapes += shape.layers != 0
			              ? fmt::format(" {}x{}x{}", shape.width, shape.height, shape.layers)
			              : fmt::format(" {}x{}", shape.width, shape.height);
		}
		for (const auto& range: state->ranges) {
			shapes += fmt::format(" 0x{:012x}+0x{:x}", range.address, range.size);
		}
		LOGF("Trace image: following the images of these shapes and ranges:%s "
		     "(KYTY_TRACE_IMAGES), up to %u snapshots (KYTY_TRACE_IMAGES_SNAPSHOTS)\n",
		     shapes.empty()
		         ? " none (the format is <width>x<height>x<layers> or 0x<address>+<size>,...)"
		         : shapes.c_str(),
		     state->max_snapshots);
		if (state->shapes.empty() && state->ranges.empty()) {
			return nullptr;
		}
		return state;
	}
};

// KYTY_TEXTURE_CACHE_MB / KYTY_BUFFER_CACHE_MB: an upper bound for a cache's collection thresholds,
// whatever the card's size (default 6 GiB each; 0 = no bound). The thresholds otherwise follow the
// device budget, and on a 32 GB card the caches kept everything up to ~25 GB: textures and buffers
// of scenes the game had left stayed until the card was full (a new game's prologue, 2026-10-07).
static uint64_t CacheCapBytes(const char* name) {
	const char* value = std::getenv(name);
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 6144ull;
	return mib == 0 ? UINT64_MAX : mib << 20u;
}

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetStreamRing()), m_buffer_cache(buffer_cache),
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
	if (const auto cap = CacheCapBytes("KYTY_TEXTURE_CACHE_MB"); cap != UINT64_MAX) {
		m_pressure_gc_memory = std::min(m_pressure_gc_memory, cap);
		m_critical_gc_memory = std::min(m_critical_gc_memory, cap + cap / 4);
		m_trigger_gc_memory  = std::min(m_trigger_gc_memory, cap / 4 * 3);
		LOGF("TextureCache: thresholds capped at %" PRIu64 " MB (trigger %" PRIu64 ", critical %" PRIu64
		     " MB)\n",
		     m_pressure_gc_memory >> 20u, m_trigger_gc_memory >> 20u, m_critical_gc_memory >> 20u);
	}
	m_trace     = TraceState::FromEnvironment();
	m_nan_check = NanCheck::FromEnvironment();
}

namespace {
// The light grid finds' background thread is stopped (CheckLightGridUpload).
void StopLightGridRechecks();
} // namespace

TextureCache::~TextureCache() {
	StopLightGridRechecks();
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

namespace {

// Images registered and unregistered (round 78): GT7 makes and drops hundreds of images a frame,
// and each change makes texture bindings kept for reuse look again (RangeStamp). Every 10 s an
// "Image churn" line: how many, where (by 256 MB area of guest memory), what dropped them
// (FreeImage's reasons) and the images made again most often (the same address, size and format).
struct ImageChurnStats {
	std::mutex                      mutex;
	uint64_t                        registered   = 0;
	uint64_t                        unregistered = 0;
	uint64_t                        bytes        = 0;
	std::map<uint64_t, uint64_t>    by_area; // address >> 28: registrations
	std::map<const char*, uint64_t> freed;   // FreeImage's reason
	struct Remade {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint32_t width   = 0;
		uint32_t height  = 0;
		uint32_t layers  = 0;
		uint32_t mips    = 0;
		int      format  = 0;
		uint64_t count   = 0;
	};
	std::unordered_map<uint64_t, Remade>  remade; // by address, size and format
	std::chrono::steady_clock::time_point last {};
};

ImageChurnStats& ImageChurn() {
	static ImageChurnStats stats;
	return stats;
}

void ReportImageChurnLocked(ImageChurnStats& stats, std::chrono::steady_clock::time_point now) {
	const double seconds = std::chrono::duration<double>(now - stats.last).count();
	std::string  areas;
	std::vector<std::pair<uint64_t, uint64_t>> by_area(stats.by_area.begin(), stats.by_area.end());
	std::ranges::sort(by_area, [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < by_area.size() && i < 6; i++) {
		areas += fmt::format("{}0x{:x}: {}", i == 0 ? "" : ", ", by_area[i].first << 28u,
		                     by_area[i].second);
	}
	std::string                                   freed;
	std::vector<std::pair<const char*, uint64_t>> reasons(stats.freed.begin(), stats.freed.end());
	std::ranges::sort(reasons, [](const auto& a, const auto& b) { return a.second > b.second; });
	for (size_t i = 0; i < reasons.size() && i < 6; i++) {
		freed += fmt::format("{}{} {}", i == 0 ? "" : ", ",
		                     reasons[i].first != nullptr ? reasons[i].first : "(other)",
		                     reasons[i].second);
	}
	std::vector<ImageChurnStats::Remade> remade;
	for (const auto& [key, entry]: stats.remade) {
		(void)key;
		if (entry.count >= 2) {
			remade.push_back(entry);
		}
	}
	std::ranges::sort(remade, [](const auto& a, const auto& b) { return a.count > b.count; });
	std::string again;
	for (size_t i = 0; i < remade.size() && i < 8; i++) {
		const auto& e = remade[i];
		again += fmt::format("{}0x{:012x}+0x{:x} {}x{} layers {} mips {} fmt {} ({}x)",
		                     i == 0 ? "" : "; ", e.address, e.size, e.width, e.height, e.layers,
		                     e.mips, e.format, e.count);
	}
	LOGF("Image churn (%.0f s): %" PRIu64 " images registered (%" PRIu64 " MB), %" PRIu64
	     " unregistered, %zu distinct (address, size, format), %zu of them made again | by area: "
	     "%s | dropped: %s | made again most: %s\n",
	     seconds, stats.registered, stats.bytes >> 20u, stats.unregistered, stats.remade.size(),
	     remade.size(), areas.empty() ? "none" : areas.c_str(),
	     freed.empty() ? "none" : freed.c_str(), again.empty() ? "none" : again.c_str());
	stats.registered   = 0;
	stats.unregistered = 0;
	stats.bytes        = 0;
	stats.by_area.clear();
	stats.freed.clear();
	stats.remade.clear();
	stats.last = now;
}

// Under the texture cache's lock: counts only (ReportImageChurnIfDue logs, outside it).
void NoteImageChurn(const Image& image, bool registered) {
	auto&           stats = ImageChurn();
	std::lock_guard lock(stats.mutex);
	if (registered) {
		stats.registered++;
		stats.bytes += image.info.data.size;
		stats.by_area[image.info.data.address >> 28u]++;
		const uint64_t words[3] = {image.info.data.address, image.info.data.size,
		                           static_cast<uint64_t>(image.info.pixel_format)};
		if (stats.remade.size() < 65536) {
			auto& entry = stats.remade[XXH3_64bits(words, sizeof(words))];
			entry       = {image.info.data.address,
			               image.info.data.size,
			               image.info.extent.width,
			               image.info.extent.height,
			               image.info.resources.layers,
			               image.info.resources.levels,
			               static_cast<int>(image.info.pixel_format),
			               entry.count + 1};
		}
	} else {
		stats.unregistered++;
	}
}

} // namespace

void ReportImageChurnIfDue() {
	auto&           stats = ImageChurn();
	const auto      now   = std::chrono::steady_clock::now();
	std::lock_guard lock(stats.mutex);
	if (stats.last.time_since_epoch().count() == 0) {
		stats.last = now;
		return;
	}
	if (now - stats.last >= std::chrono::seconds(10)) {
		ReportImageChurnLocked(stats, now);
	}
}

namespace {

void NoteImageFreed(const char* why) {
	auto&           stats = ImageChurn();
	std::lock_guard lock(stats.mutex);
	if (stats.freed.size() < 64 || stats.freed.contains(why)) {
		stats.freed[why]++;
	}
}

} // namespace

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	m_slot_images[id].self_id = id;
	Profiler::Add(Profiler::Counter::ImagesCreated);
	if (!info.data.Empty()) {
		RegisterImage(id);
		if (m_trace != nullptr) {
			TraceImageCreated(id);
		}
	}
	return id;
}

TextureCache::RegisteredPageCounts::~RegisteredPageCounts() {
	for (size_t index = 0; index < LeafCount; index++) {
		delete m_leaves[index].load(std::memory_order_relaxed);
	}
}

void TextureCache::RegisteredPageCounts::Add(size_t page, int32_t delta) {
	auto& slot = m_leaves[page >> LeafBits];
	auto* leaf = slot.load(std::memory_order_relaxed);
	if (leaf == nullptr) {
		leaf = new Leaf {};
		slot.store(leaf, std::memory_order_release);
	}
	(*leaf)[page & (LeafSize - 1)].fetch_add(static_cast<uint32_t>(delta),
	                                         std::memory_order_release);
}

bool TextureCache::RegisteredPageCounts::Any(size_t first, size_t last_exclusive) const noexcept {
	for (auto page = first; page < last_exclusive; page++) {
		const auto* leaf = m_leaves[page >> LeafBits].load(std::memory_order_acquire);
		if (leaf != nullptr &&
		    (*leaf)[page & (LeafSize - 1)].load(std::memory_order_acquire) != 0) {
			return true;
		}
	}
	return false;
}

bool TextureCache::MayHaveImages(uint64_t address, uint64_t size) const noexcept {
	ImagePageTable::PageRange pages {};
	return !ImagePageTable::TryGetPageRange(address, size, pages) ||
	       m_registered_pages.Any(pages.first, pages.last_exclusive);
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
		m_registered_pages.Add(page, 1);
	});
	image.registered = true;
	m_image_set_generation.fetch_add(1, std::memory_order_acq_rel);
	m_zone_stamps.Stamp(image.info.data.address, image.info.data.size);
	NoteImageChurn(image, true);
	NoteBindlessStateChange(image);
	image.lru_tick = LruClock();
	image.lru_id   = m_lru_cache.Insert(id, image.lru_tick);
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
		m_registered_pages.Add(page, -1);
	});
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
	m_image_set_generation.fetch_add(1, std::memory_order_acq_rel);
	m_zone_stamps.Stamp(image.info.data.address, image.info.data.size);
	NoteImageChurn(image, false);
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	if (image->depth_id) {
		if (auto* depth = m_slot_images.try_get(image->depth_id); depth != nullptr) {
			std::erase(depth->stencil_ids, id);
		}
	} else {
		// Taken first: deleting each association unlinks it from this list too.
		for (const auto association: std::exchange(image->stencil_ids, {})) {
			if (const auto* associated = m_slot_images.try_get(association);
			    associated != nullptr && associated->depth_id == id) {
				FreeImage(association);
			}
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	if (image->read_only_alias) {
		image->read_only_alias = false;
		m_read_only_aliases--;
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
	// Also between guest command buffers (a guest unmap runs there): command buffers submitted
	// earlier may still be sampling the image on the GPU. Destroying it at once let them read
	// freed memory (device loss at the title, 2026-10-04).
	m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
}

void TextureCache::ReleaseImage(ImageId id) {
	const auto* image = m_slot_images.try_get(id);
	if (image != nullptr && on_image_release && on_image_release(*image)) {
		// Work recorded with the repointed slot runs until the current tick is done.
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	if (image != nullptr && !m_scheduler.IsFree(image->tick_accessed_last)) {
		// Used after it was dropped, in a tick the GPU has not finished: keep it until then.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("TextureCache: image %ux%u guest=0x%016" PRIx64 " used in tick %" PRIu64
			     " after its release was queued; destruction waits for that tick\n",
			     image->info.extent.width, image->info.extent.height, image->info.data.address,
			     image->tick_accessed_last);
		}
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	m_slot_images.erase(id);
	Profiler::Add(Profiler::Counter::ImagesDestroyed);
}

void TextureCache::FreeImage(ImageId id, const char* why, bool copied) {
	auto& image = m_slot_images[id];
	if (image.registered) {
		NoteImageFreed(why);
	}
	// Pixels only this image held: the next image over the memory starts from the guest bytes.
	if (image.gpu_pixels && image.IsGpuModified() && !copied && why != nullptr &&
	    image.registered && !image.depth_id) {
		static std::atomic<uint64_t> dropped {0};
		const auto                   count = ++dropped;
		if (count <= 32 || std::has_single_bit(count)) {
			LOGF("TextureCache: GPU-written image 0x%012" PRIx64 "+0x%" PRIx64
			     " %ux%u layers %u mips %u fmt %d tile %u dropped without a write-back (%s): the "
			     "next image over its memory starts from the guest bytes (%" PRIu64 " so far)\n",
			     image.info.data.address, image.info.data.size, image.info.extent.width,
			     image.info.extent.height, image.info.resources.layers, image.info.resources.levels,
			     static_cast<int>(image.info.pixel_format),
			     static_cast<uint32_t>(image.info.tile_mode), why, count);
		}
	}
	if (TracesImage(id)) {
		TraceImageEvent(id,
		                fmt::format("freed: {}{}", why != nullptr ? why : "(no reason given)",
		                            !(image.gpu_pixels && image.IsGpuModified()) ? ""
		                            : copied ? "; its GPU pixels were kept"
		                                     : "; its GPU pixels are lost"),
		                false);
		m_trace->images.erase(id);
	}
	if (image.IsGpuModified()) {
		image.ClearGpuModified();
	}
	// Its copies (SingleSampleDepth, SingleSampleStencil, SubExtentCopy) go with it.
	for (auto* copies: {&m_single_sample_depths, &m_single_sample_stencils}) {
		if (const auto copy = copies->find(id); copy != copies->end()) {
			const auto copy_id = copy->second.copy;
			copies->erase(copy);
			if (copy_id) {
				m_scheduler.DeferRelease([this, copy_id] { ReleaseImage(copy_id); });
			}
		}
	}
	if (const auto copies = m_sub_extent_copies.find(id); copies != m_sub_extent_copies.end()) {
		for (const auto& entry: copies->second) {
			if (const auto copy_id = entry.copy; copy_id) {
				m_scheduler.DeferRelease([this, copy_id] { ReleaseImage(copy_id); });
			}
		}
		m_sub_extent_copies.erase(copies);
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	const auto frame          = m_graphics.presented_frames.load(std::memory_order_relaxed);
	image.frame_accessed_last = frame;
	if (image.registered) {
		// LruClock, from the frame already read.
		const auto clock = frame + m_gc_tick / 512;
		if (image.lru_tick < clock) {
			image.lru_tick = clock;
			m_lru_cache.Touch(image.lru_id, clock);
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

// KYTY_READBACK_SMALL_IMAGES_KB (default 32768 since round 126, 64 before; 0: off): GPU-written
// linear images up to this size are copied back to guest memory after each submission, as
// --readback-linear-images does for
// every linear image. A small linear target or storage image nothing on the GPU reads is how a
// game reads GPU results on the CPU: GT7 writes 16x16 and 32x32 reductions of its scene colour
// and depth each frame for the CPU, which sets the white balance (and more) from them. Without
// this the CPU read stale bytes; small render targets are never evicted, so never written back.
// Round 126: a livery's pictures are drawn into a 1920x540 linear target (4 MB) twice, the halves
// of the 1920x1080 picture the CPU encodes as PNG; with 64 KB that picture was noise, with 16 MB
// right. 32 MB holds a 3840x2160 RGBA8 picture (a photo). The frames of a race write none above
// 64 KB, so they read back what they did before.
static uint64_t SmallReadbackBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_READBACK_SMALL_IMAGES_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 32768ull) << 10u;
	}();
	return bytes;
}

namespace {

// Round 120 (KYTY_READBACK_PREFILL, default on; 0 turns it off). A read-back for the CPU lands in
// guest memory once the host GPU has run the submission that drew it, but the guest may see the
// GPU finish before that: labels and interrupts are written as they are recorded. A game thread
// that reads the result at once takes what the memory held before. A game that takes its
// read-backs from a ring of per-frame memory finds another kind of read-back there as often as
// not (depth floats read as colour halves, values in the tens of thousands): its white balance,
// set from a 16x16 reduction of the scene colour, went to its limit in tunnels, and the picture
// turned green. So when a read-back is enrolled (its draw recorded, before any later label), its
// guest memory gets the bytes the last read-back of its kind landed: a reader that comes early
// takes the frame before's result instead. A kind is a format, a size, and the order among the
// read-backs of that format and size in a frame (a game may make two of them a frame).
struct ReadbackKind {
	int32_t  format = 0;
	uint32_t width  = 0;
	uint32_t height = 0;
	uint64_t size   = 0;
	uint32_t index  = 0;

	[[nodiscard]] bool operator<(const ReadbackKind& other) const noexcept {
		return std::tie(format, width, height, size, index) <
		       std::tie(other.format, other.width, other.height, other.size, other.index);
	}
	[[nodiscard]] bool operator==(const ReadbackKind& other) const noexcept {
		return format == other.format && width == other.width && height == other.height &&
		       size == other.size && index == other.index;
	}
};

struct ReadbackPrefill {
	std::mutex mutex;
	// The GPU thread (enrolment) and the priority thread (landing), under the mutex.
	std::map<ReadbackKind, std::vector<uint8_t>> latest;      // the last bytes each kind landed
	std::map<uint64_t, ReadbackKind>             landed_kind; // the kind that last landed there
	struct Enrolled {
		ReadbackKind kind;
		uint64_t     ns = 0; // when its draw was recorded
	};
	std::map<uint64_t, Enrolled> enrolled;
	// The GPU thread only: the read-backs of this frame, and how many of each format and size.
	uint64_t                         frame = UINT64_MAX;
	std::unordered_set<uint64_t>     frame_addresses;
	std::map<ReadbackKind, uint32_t> frame_counts;
	// Statistics (under the mutex), logged every 10 s.
	uint64_t since_ns    = 0;
	uint64_t enrolments  = 0;
	uint64_t other_kind  = 0; // held a read-back of another kind
	uint64_t other_bytes = 0; // held bytes no read-back landed
	uint64_t older_own   = 0; // held an older read-back of its kind
	uint64_t already     = 0; // held the latest of its kind
	uint64_t first       = 0; // no read-back of its kind had landed yet
	uint64_t written     = 0;
	uint64_t landed      = 0;
	uint64_t landed_ns   = 0;
	uint64_t longest_ns  = 0;
	uint64_t dropped     = 0;
};

ReadbackPrefill& GetReadbackPrefill() {
	// Never destroyed: read-backs may still land during static destruction.
	static auto* prefill = new ReadbackPrefill;
	return *prefill;
}

bool ReadbackPrefillEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_PREFILL");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("Read-backs for the CPU: %s (KYTY_READBACK_PREFILL)\n",
		     on ? "until one lands, its guest memory holds the last of its kind"
		        : "guest memory keeps what it held until a read-back lands");
		return on;
	}();
	return enabled;
}

// Under the mutex: every 10 s, what the read-backs' guest memory held when they were enrolled.
void LogReadbackPrefill(ReadbackPrefill& prefill, uint64_t now_ns) {
	if (prefill.since_ns == 0) {
		prefill.since_ns = now_ns;
		return;
	}
	if (now_ns - prefill.since_ns < 10'000'000'000ull) {
		return;
	}
	const bool on = ReadbackPrefillEnabled();
	LOGF("Read-backs for the CPU (%.0f s): %" PRIu64 " new; before they landed, their memory held "
	     "another kind of read-back in %" PRIu64 ", bytes no read-back left in %" PRIu64
	     ", an older one of their kind in %" PRIu64 " (%s %" PRIu64
	     " of these the last bytes of their kind), the latest of their kind in %" PRIu64
	     ", and %" PRIu64 " were the first of their kind; %" PRIu64
	     " landed, %.2f ms after their draw on average (longest %.2f ms), %" PRIu64
	     " dropped (the guest wrote there first)\n",
	     static_cast<double>(now_ns - prefill.since_ns) / 1e9, prefill.enrolments,
	     prefill.other_kind, prefill.other_bytes, prefill.older_own,
	     on ? "gave" : "would have given (KYTY_READBACK_PREFILL=0)", prefill.written,
	     prefill.already, prefill.first, prefill.landed,
	     prefill.landed != 0
	         ? static_cast<double>(prefill.landed_ns) / 1e6 / static_cast<double>(prefill.landed)
	         : 0.0,
	     static_cast<double>(prefill.longest_ns) / 1e6, prefill.dropped);
	prefill.since_ns    = now_ns;
	prefill.enrolments  = 0;
	prefill.other_kind  = 0;
	prefill.other_bytes = 0;
	prefill.older_own   = 0;
	prefill.already     = 0;
	prefill.first       = 0;
	prefill.written     = 0;
	prefill.landed      = 0;
	prefill.landed_ns   = 0;
	prefill.longest_ns  = 0;
	prefill.dropped     = 0;
}

// GPU thread: a small read-back enrolled in `frame` (presented frames so far).
void PrefillReadback(const ImageInfo& info, uint64_t frame) {
	const auto range = info.data;
	if (range.Empty() || range.size > SmallReadbackBytes() || info.samples != 1) {
		return;
	}
	(void)ReadbackPrefillEnabled(); // logs the setting at the first read-back
	auto& prefill = GetReadbackPrefill();
	if (prefill.frame != frame) {
		prefill.frame = frame;
		prefill.frame_addresses.clear();
		prefill.frame_counts.clear();
	}
	if (!prefill.frame_addresses.insert(range.address).second) {
		// Drawn again in a later submission of the frame: its first read-back may have landed.
		return;
	}
	ReadbackKind kind;
	kind.format = static_cast<int32_t>(info.pixel_format);
	kind.width  = info.extent.width;
	kind.height = info.extent.height;
	kind.size   = range.size;
	kind.index  = prefill.frame_counts[kind]++;

	std::vector<uint8_t> current(static_cast<size_t>(range.size));
	const bool           readable =
	    LibKernel::Memory::TryReadBacking(range.address, current.data(), range.size);
	const auto own_hash = LandedReadbackHash(range.address);
	const auto now_ns   = LibKernel::Memory::JournalNowNs();

	std::lock_guard lock(prefill.mutex);
	prefill.enrolments++;
	prefill.enrolled[range.address] = {kind, now_ns};
	LogReadbackPrefill(prefill, now_ns);
	const auto latest = prefill.latest.find(kind);
	if (!readable || latest == prefill.latest.end() || latest->second.size() != current.size()) {
		prefill.first++;
		return;
	}
	if (current == latest->second) {
		prefill.already++;
		return;
	}
	const auto landed = prefill.landed_kind.find(range.address);
	if (own_hash == 0 || landed == prefill.landed_kind.end() ||
	    HashBytes(current.data(), current.size()) != own_hash) {
		prefill.other_bytes++;
	} else if (landed->second.format != kind.format || landed->second.size != kind.size ||
	           landed->second.width != kind.width || landed->second.height != kind.height) {
		prefill.other_kind++;
	} else {
		prefill.older_own++;
	}
	prefill.written++;
	if (ReadbackPrefillEnabled()) {
		const LibKernel::Memory::WriteSourceScope source(
		    LibKernel::Memory::WriteSource::ImageWriteBack);
		(void)LibKernel::Memory::TryWriteBacking(range.address, latest->second.data(),
		                                         latest->second.size());
	}
}

// Priority thread: a small read-back landed (its bytes are in guest memory now), or was dropped.
void NoteReadbackLanded(uint64_t address, const uint8_t* bytes, uint64_t size, bool dropped) {
	auto&           prefill = GetReadbackPrefill();
	const auto      now_ns  = LibKernel::Memory::JournalNowNs();
	std::lock_guard lock(prefill.mutex);
	const auto      enrolled = prefill.enrolled.find(address);
	if (enrolled == prefill.enrolled.end() || enrolled->second.kind.size != size) {
		return;
	}
	if (dropped) {
		prefill.dropped++;
	} else {
		prefill.latest[enrolled->second.kind].assign(bytes, bytes + size);
		prefill.landed_kind[address] = enrolled->second.kind;
		const auto waited            = now_ns - std::min(now_ns, enrolled->second.ns);
		prefill.landed++;
		prefill.landed_ns += waited;
		prefill.longest_ns = std::max(prefill.longest_ns, waited);
	}
	prefill.enrolled.erase(enrolled);
}

// Guest memory unmapped.
void ForgetReadbackPrefill(uint64_t address, uint64_t size) {
	auto&           prefill = GetReadbackPrefill();
	std::lock_guard lock(prefill.mutex);
	prefill.landed_kind.erase(prefill.landed_kind.lower_bound(address),
	                          prefill.landed_kind.lower_bound(address + size));
	prefill.enrolled.erase(prefill.enrolled.lower_bound(address),
	                       prefill.enrolled.lower_bound(address + size));
}

} // namespace

bool TextureCache::ReadsBackForCpu(const ImageInfo& info) const {
	return !info.IsTiled() && !info.data.Empty() &&
	       (m_readback_linear_images ||
	        (info.data.size <= SmallReadbackBytes() && info.samples == 1));
}

std::vector<std::string> TextureCache::DescribeImagesInRange(uint64_t address, uint64_t size,
                                                             size_t count) {
	std::vector<std::string> lines;
	std::scoped_lock         lock {m_lock};
	const auto               ids = FindImagesInRegion(address, size, false);
	for (const auto id: ids) {
		if (lines.size() >= count) {
			break;
		}
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr) {
			continue;
		}
		const auto& info = image->info;
		lines.push_back(fmt::format(
		    "0x{:012x}+0x{:x} {}x{}x{} layers {} mips {} fmt {} guest {} tile {} samples "
		    "{}{}{}{}{}{}, {}",
		    info.data.address, info.data.size, info.extent.width, info.extent.height,
		    info.extent.depth, info.resources.layers, info.resources.levels,
		    static_cast<int>(info.pixel_format), static_cast<uint32_t>(info.guest_format),
		    static_cast<uint32_t>(info.tile_mode), info.samples,
		    image->registered ? "" : ", unregistered",
		    image->IsGpuModified() ? ", GPU-written" : "", image->gpu_pixels ? ", GPU pixels" : "",
		    image->IsCpuDirty() ? ", CPU-dirty" : "",
		    image->usage.render_target ? ", a render target" : "",
		    ReadsBackForCpu(info) ? "read back for the CPU after each submission"
		                          : "not read back for the CPU after each submission"));
	}
	if (ids.size() > lines.size()) {
		lines.push_back(fmt::format("{} more", ids.size() - lines.size()));
	}
	return lines;
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	if (!ReadsBackForCpu(image.info)) {
		return;
	}
	if (!image.IsGpuModified()) {
		EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
	}
	const bool added = m_download_images.insert(id).second;
	if (added) {
		PrefillReadback(image.info, m_graphics.presented_frames.load(std::memory_order_relaxed));
	}
	if (!m_readback_linear_images) {
		// The first images read back, each once (under m_lock).
		static std::unordered_set<uint64_t> logged;
		if (logged.size() < 16 && logged.insert(image.info.data.address).second) {
			LOGF("TextureCache: read back for the CPU: 0x%012" PRIx64 "+0x%" PRIx64
			     " %ux%u format %d%s\n",
			     image.info.data.address, image.info.data.size, image.info.extent.width,
			     image.info.extent.height, static_cast<int>(image.info.pixel_format),
			     logged.size() == 1
			         ? " (GPU-written linear images up to KYTY_READBACK_SMALL_IMAGES_KB, default "
			           "32768, go back to guest memory after each submission)"
			         : "");
		}
	}
}

TextureCache::DiagnosticState TextureCache::DiagnosticImageState(ImageId id) {
	DiagnosticState  state;
	std::scoped_lock lock {m_lock};
	const auto*      image = m_slot_images.try_get(id);
	if (image == nullptr) {
		return state;
	}
	state.valid        = true;
	state.gpu_written  = image->IsGpuModified();
	state.buffer_write = image->IsBufferModified();
	state.cpu_dirty    = image->IsCpuDirty();
	state.data         = image->info.data;
	state.extent       = image->info.extent;
	state.levels       = image->info.resources.levels;
	state.layers       = image->info.resources.layers;
	state.host_width   = image->backing.extent.width;
	state.host_height  = image->backing.extent.height;
	state.format       = image->info.pixel_format;
	if (!image->registered || image->info.data.Empty()) {
		return state;
	}
	FindImagesInRegion(image->info.data.address, image->info.data.size, false)
	    .ForEach([&](ImageId other) {
		    const auto* overlap = other != id ? m_slot_images.try_get(other) : nullptr;
		    if (overlap == nullptr) {
			    return;
		    }
		    if (state.overlaps++ == 0) {
			    state.first_overlap = overlap->info.data.address;
		    }
		    state.gpu_overlaps += overlap->IsGpuModified() ? 1u : 0u;
	    });
	return state;
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
		destination.MarkGpuModified();
		destination.gpu_pixels = destination.gpu_pixels || source.gpu_pixels;
	}
	destination.ClearBufferModified();
	destination.content_serial++;
	if (m_trace != nullptr) {
		TraceImageChanged(destination, "copy from an image over its memory");
	}
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
		destination.gpu_pixels = destination.gpu_pixels || source.gpu_pixels;
	}
	destination.content_serial++;
	if (m_trace != nullptr) {
		TraceImageChanged(destination, "copy from an image over its memory");
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
		info.mip_layout = cached.info.mip_layout;
	} else {
		info.resources = std::max(requested.resources, cached.info.resources);
		// Round 123: the layers of the image it replaces, in a range that holds them. Before,
		// a depth binding of one slice over a depth image of four (the livery editor saving a
		// livery) made an image of four layers in the range of one, and its first upload ended
		// the emulator in the tiler (a 512x512 32-bit slice in 256 KB).
		if (info.resources != requested.resources &&
		    !GrowDepthLayers(info, requested, cached.info)) {
			info.resources = requested.resources;
		}
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	bool copied = false;
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
			copied = true;
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
		copied = true;
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id, "replaced by an image of another kind over the same memory", copied);
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
	const auto current_tick  = m_scheduler.CurrentTick();
	const auto current_frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	const bool safe_to_delete =
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	    (!AliasAgeByFrames() || current_frame - std::min(current_frame, cached.frame_accessed_last) >
	                                AliasFramesBeforeRemoval);

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (KeepReadOnlyAlias(requested, binding, cached)) {
			return {.image = merged_id, .kept_alias = true};
		}
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     requested.mip_layout != cached.info.mip_layout)) {
			if (safe_to_delete) {
				FreeImage(cached_id,
				          "an image of another tile mode or mip layout took its address");
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// A 1x1x1 texel read as 2D by one shader and 3D by another has the same guest bytes in
		// either type: keep both native images instead of recreating one per request (a menu
		// did this ~110 times a frame). Only for clean, untouched-by-GPU single texels.
		if (requested.IsVolume() != cached.info.IsVolume() && requested.type != cached.info.type &&
		    requested.data.size == cached.info.data.size && requested.extent.width == 1 &&
		    requested.extent.height == 1 && requested.extent.depth == 1 &&
		    cached.info.extent.width == 1 && cached.info.extent.height == 1 &&
		    cached.info.extent.depth == 1 && requested.resources == cached.info.resources &&
		    requested.resources.levels == 1 && requested.resources.layers == 1 &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    !cached.IsGpuModified() && !cached.binding.is_target && [&] {
			    size_t owners = 0;
			    ForEachPage(requested.data.address, 1, [&](uint64_t page) {
				    if (const auto* list = m_image_page_table.Find(page)) {
					    owners = list->size();
				    }
			    });
			    return owners < 4;
		    }()) {
			// Compatible formats only: an incompatible one still goes through the replace path
			// below, which frees the old image. Keeping every format/type pair alive piled
			// >127 owners on this page (PageManager write-watcher overflow).
			return {merged_id};
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
		// A texture smaller than the image with the same layout is viewed through the image here;
		// its binding reads a copy of the image's corner (SubExtentCopy, round 124).
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
		FreeImage(cached_id, "merged into a larger image", true);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id, "an image over its memory took its place");
	}
	return {merged_id};
}

namespace {
// Read-only aliases (round 79): at most this large, each.
constexpr uint64_t ReadOnlyAliasMaxBytes = uint64_t {64} << 10u;
} // namespace

bool TextureCache::ReadOnlyAliasCandidate(const Image& image) {
	return image.registered && !image.depth_id && image.stencil_ids.empty() &&
	       !image.IsGpuModified() && !image.gpu_pixels && !image.IsBufferModified() &&
	       !image.usage.storage && !image.usage.render_target && !image.usage.depth_target &&
	       !image.usage.video_out && !image.binding.is_target && !image.binding.shader_write &&
	       image.info.samples == 1 && image.backing.samples == 1 && !image.info.HasStencil() &&
	       !image.info.HasMetadata() && !image.info.data.Empty() &&
	       image.info.data.size <= ReadOnlyAliasMaxBytes;
}

bool TextureCache::KeepReadOnlyAlias(const ImageInfo& requested, BindingType binding,
                                     const Image& cached) const {
	if (binding != BindingType::Texture || Bisect::ReadOnlyAliasesOff() ||
	    !ReadOnlyAliasCandidate(cached) || requested.samples != 1 || requested.HasStencil() ||
	    requested.HasMetadata() || requested.data.Empty() ||
	    requested.data.size > ReadOnlyAliasMaxBytes ||
	    requested.data.address != cached.info.data.address ||
	    requested.tile_mode != cached.info.tile_mode) {
		return false;
	}
	// Kinds one native image cannot serve both of, where a read of the other kind replaced the
	// image before: a volume and a flat image (ResolveDepthOverlap's recreation, ExpandImage), a
	// depth read of a color image (ResolveDepthOverlap's recreation). A color read of a depth
	// image goes on using the depth image, as before.
	return requested.IsVolume() != cached.info.IsVolume() ||
	       (requested.IsDepth() && !cached.info.IsDepth());
}

void TextureCache::MarkReadOnlyAlias(Image& image) {
	if (!image.read_only_alias && image.registered) {
		image.read_only_alias = true;
		m_read_only_aliases++;
	}
}

void TextureCache::FreeReadOnlyAliasesOf(const Image& image) {
	if (image.info.data.Empty()) {
		return;
	}
	// Its siblings start where it does: they are among the owners of its first page.
	ImageIds siblings;
	if (const auto* owners =
	        m_image_page_table.Find(image.info.data.address >> ImagePageTable::kPageBits);
	    owners != nullptr) {
		owners->ForEach([&](ImageId id) {
			const auto* other = m_slot_images.try_get(id);
			// One bound as a target in the draw at hand is left to its own write (freeing it
			// here would leave the draw a target that is gone).
			if (other != nullptr && other != &image && other->registered &&
			    other->read_only_alias && other->info.data.address == image.info.data.address &&
			    !other->IsGpuModified() && !other->binding.is_target) {
				siblings.push_back(id);
			}
		});
	}
	siblings.ForEach([&](ImageId id) {
		auto& other = m_slot_images[id];
		if (other.binding.is_bound || other.binding.is_target) {
			other.binding.needs_rebind = true;
		}
		FreeImage(id, "a GPU write over a read-only alias of it");
	});
	// Written by the GPU now: no longer one either.
	if (auto* self = m_slot_images.try_get(image.self_id);
	    self != nullptr && self == &image && self->read_only_alias) {
		self->read_only_alias = false;
		m_read_only_aliases--;
	}
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
	FreeImage(source_id, "expanded into a larger image", true);
	return expanded_id;
}

struct TextureCache::TextureTransfer {
	TextureUploadLayout              layout;
	std::vector<vk::BufferImageCopy> regions;
	std::vector<GpuTileInfo>         tiles;
	ColorTransform                   color_transform = ColorTransform::None;
	bool                             valid           = false;

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
	transfer.color_transform = info.GetColorTransform();
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
		// Round 86: a depth image whose rows (pitch x height) do not fit in its share of its
		// range is not downloaded (DownloadDepth stopped the emulator on one that the garbage
		// collector wrote back, in another title of the series). It stays until it is safe to
		// free, as an image whose download is unsupported does.
		if (transfer.valid) {
			const uint64_t guest_slice =
			    static_cast<uint64_t>(info.pitch) * info.extent.height * info.bytes_per_block;
			transfer.valid = guest_slice <= info.data.size / info.resources.layers;
		}
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
			                        transfer.LinearSize(), transfer.tiles, transfer.color_transform);
		} else if (transfer.color_transform != ColorTransform::None) {
			linear = m_tiler.TransformColor(linear, transfer.color_transform, true);
		}
		upload(transfer.regions, linear);
		return;
	}

	// The stencil plane has its own row pitch and shares the native image
	// with the depth plane.
	auto info = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
		info.resources       = {image.stencil_subresources.level_count,
		                        image.stencil_subresources.layer_count};
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
	// Round 123: a slice its range cannot hold (an image made with more layers than its range
	// has room for) is left as it is rather than ending the emulator in the tiler.
	if (const uint64_t slice_bytes =
	        static_cast<uint64_t>(info.pitch) * info.extent.height * info.bytes_per_block;
	    info.extent.height == 0 || slice_bytes > full_slice_size) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("TextureCache: depth %s 0x%012" PRIx64 "+0x%" PRIx64
			     " %ux%u pitch %u, %u bytes a texel, %u layers: a slice needs 0x%" PRIx64
			     " bytes and has 0x%" PRIx64 "; not uploaded (round 123)\n",
			     image.depth_id ? "stencil" : "image", info.data.address, info.data.size,
			     info.extent.width, info.extent.height, info.pitch, info.bytes_per_block, layers,
			     slice_bytes, full_slice_size);
		}
		return;
	}
	auto copies = BuildDepthCopies(info, full_slice_size, image.depth_id
	                                                        ? vk::ImageAspectFlagBits::eStencil
	                                                        : vk::ImageAspectFlagBits::eDepth);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
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
		if (!image.initialized && Bisect::ZeroNewMemory()) {
			ZeroNewImage(id);
		}
		image.initialized = true;
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		if (!image.initialized && Bisect::ZeroNewMemory()) {
			ZeroNewImage(id);
		}
		image.initialized = true;
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload && !image.initialized && ClearFromKnownFill(id)) {
		image.initialized = true;
		return;
	}
	if (upload) {
		const auto range         = image.info.data;
		Buffer*    source        = nullptr;
		uint64_t   source_offset = 0;
		// The GPU's pixels give way to the guest bytes (or to a cached buffer's, which hold a GPU
		// buffer write): right when the CPU (or that write) replaced all of them, not when it
		// wrote only some.
		const bool over_gpu_pixels = image.gpu_pixels && image.IsGpuModified();
		static std::atomic<uint64_t> replaced {0};
		const auto                   count = over_gpu_pixels ? ++replaced : 0;
		const bool log_replaced = over_gpu_pixels && (count <= 32 || std::has_single_bit(count));
		if (log_replaced || TracesImage(id)) {
			const bool gpu_bytes = m_buffer_cache.IsRegionGpuModified(range.address, range.size);
			const char* cpu       = image.IsDefinitelyCpuDirty() ? "a CPU write"
			                        : image.IsMaybeCpuDirty()    ? "a CPU write next to it"
			                                                     : "";
			const auto  reason    = !image.initialized
			                            ? std::string("its creation")
			                            : fmt::format("{}{}", cpu,
                                                  image.IsBufferModified()
			                                              ? (*cpu != '\0' ? " and a GPU buffer write"
			                                                              : "a GPU buffer write")
			                                              : "");
			const auto from =
			    gpu_bytes ? "a cached buffer that holds GPU-written bytes" : "the guest bytes";
			if (log_replaced) {
				LOGF("TextureCache: GPU-written image 0x%012" PRIx64 "+0x%" PRIx64
				     " %ux%u layers %u mips %u fmt %d tile %u reloaded from %s after %s: what the "
				     "GPU wrote into it is replaced (%" PRIu64 " so far)\n",
				     range.address, range.size, image.info.extent.width, image.info.extent.height,
				     image.info.resources.layers, image.info.resources.levels,
				     static_cast<int>(image.info.pixel_format),
				     static_cast<uint32_t>(image.info.tile_mode), from, reason.c_str(), count);
			}
			if (TracesImage(id)) {
				TraceImageEvent(id,
				                fmt::format("uploaded from {} after {}{}", from, reason,
				                            over_gpu_pixels ? ", over its GPU pixels" : ""),
				                true);
				TraceImageChanged(image, "upload");
			}
		}
		// Hashed before it is staged: a CPU write between the two leaves the hash older than
		// the upload, so the next comparison uploads once more rather than missing it.
		uint64_t hash = 0;
		if (ChecksGuestContent(image) && HashGuestBacking(range.address, range.size, hash)) {
			std::tie(source, source_offset) =
			    m_buffer_cache.StageGuestBytesForImage(range.address, range.size);
			image.content_hash             = hash;
			image.content_hash_valid       = true;
			image.content_check_submission = g_guest_submission_seq.load(std::memory_order_relaxed);
		} else {
			std::tie(source, source_offset) =
			    m_buffer_cache.ObtainBufferForImage(range.address, range.size);
			image.content_hash_valid = false;
		}
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		CheckLightGridUpload(image, *source, source_offset);
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
		image.gpu_pixels = false;
		image.content_serial++;
		if (TracesImage(id)) {
			TraceUploadScan(id);
		}
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	image.initialized = true;
}

// KYTY_GPU_CLEAR_RECHECK=1, or KYTY_BISECT's recheck: a fast clear is checked on the GPU again at
// the first binding of each command buffer even when nothing wrote its metadata since the last
// check, as before round 65.
bool TextureCache::RechecksColorClears() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_GPU_CLEAR_RECHECK");
		return value != nullptr && value[0] == '1';
	}();
	return always || Bisect::GpuClearRecheck();
}

bool TextureCache::MaterializeColorClearOnGpu(ImageId id, const ImageDesc& desc, uint32_t first,
                                              uint32_t image_first, uint32_t count,
                                              uint64_t slice_size, bool unwritten,
                                              uint32_t* check) {
	const auto& view    = desc.view_info;
	const auto  decline = [&](const char* why) {
		// Once per reason: the metadata is then read back, draining the GPU.
		static std::mutex                   logged_mutex;
		static std::unordered_set<std::string> logged;
		std::lock_guard                     lock(logged_mutex);
		if (logged.insert(why).second) {
			LOGF("TextureCache: color clear of 0x%016" PRIx64
			     " checked by a read-back: %s (binding %u, samples %u, format %u)\n",
			     desc.info.data.address, why, static_cast<uint32_t>(desc.type),
			     desc.info.samples, static_cast<uint32_t>(view.format));
		}
		return false;
	};
	// A texture or storage binding of a target is checked the same way: once the render target's
	// check left its metadata GPU-written, the first sampling of it downloaded it instead. And
	// (round 80) a flip's video-out read: the game's display buffers' DCC is rewritten every frame
	// by its last compute dispatch, right before the flip, and reading it back waited ~3 ms a frame
	// for the GPU to finish the frame (round 79's races; found no clear there: the menus' black
	// frames are where it does). KYTY_BISECT's flipcheck:old reads it back, as before.
	if (!GpuColorClearEnabled() || (desc.type == BindingType::VideoOut && Bisect::FlipCheckOld())) {
		return false;
	}
	if (desc.info.IsVolume()) {
		return decline("volume target");
	}
	if (desc.type != BindingType::RenderTarget &&
	    !(m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
	      vk::FormatFeatureFlagBits::eColorAttachment)) {
		return decline("texture view format is not renderable");
	}
	if (slice_size == 0 || slice_size % sizeof(uint32_t) != 0 ||
	    slice_size / sizeof(uint32_t) > UINT32_MAX) {
		return decline("metadata slice size");
	}
	// The values the metadata may hold a clear of (DecodeColorClear's codes).
	std::vector<ConditionalColorClear::Candidate> candidates;
	for (const uint8_t code: {uint8_t {0x00}, uint8_t {0x20}, uint8_t {0x40}, uint8_t {0x80},
	                          uint8_t {0xc0}}) {
		vk::ClearColorValue clear {};
		if (DecodeColorClear(desc, code, clear)) {
			ConditionalColorClear::Candidate candidate {.code = code};
			std::memcpy(candidate.bits.data(), clear.uint32.data(), sizeof(candidate.bits));
			candidates.push_back(candidate);
		}
	}
	if (candidates.empty()) {
		return true; // no code this target could be cleared with: nothing to check
	}
	uint32_t samples = 1;
	{
		std::scoped_lock lock {m_lock};
		const auto&      image = m_slot_images[id];
		if (image.backing.image == nullptr) {
			return decline("no image yet");
		}
		if (!(image.backing.usage & vk::ImageUsageFlagBits::eColorAttachment)) {
			return decline("image without color-attachment use");
		}
		// A flip of guest bytes newer than the image goes the old way: the read-back and its
		// check that a compressed one has none (FindImage). (A CPU write next to it only, a
		// "maybe", is checked against its hash by InitializeImage below.)
		if (desc.type == BindingType::VideoOut &&
		    (image.IsBufferModified() || image.IsDefinitelyCpuDirty())) {
			return decline("video-out image older than its guest bytes");
		}
		samples = image.backing.samples;
	}
	if (m_conditional_clear == nullptr) {
		m_conditional_clear = std::make_unique<ConditionalColorClear>(m_graphics, m_scheduler);
		LOGF("TextureCache: render-target fast clears are checked on the GPU "
		     "(KYTY_GPU_COLOR_CLEAR=0 reads the metadata back)\n");
	}
	const auto& range = desc.info.metadata.range;
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		// The GPU copy of the slice's keys (read and rewritten by the check), synchronized first.
		const auto [buffer, offset] =
		    m_buffer_cache.ObtainBuffer(address, slice_size, true, false, {}, true);
		EXIT_IF(buffer == nullptr || !buffer->HasDeviceAddress());
		std::scoped_lock lock {m_lock};
		auto&            image = m_slot_images[id];
		TrackImage(id);
		if (image.IsBufferModified() || image.IsCpuDirty()) {
			// Drawn over in part, if at all: the guest contents come first.
			InitializeImage(id);
		}
		ImageViewInfo target {};
		target.format      = view.format;
		target.type        = vk::ImageViewType::e2D;
		target.base_level  = view.base_level;
		target.base_layer  = image_first + slice;
		target.layer_count = 1;
		target.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		auto& command      = m_scheduler.Current();
		command.EndRendering();
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite |
		                  vk::AccessFlagBits2::eColorAttachmentRead,
		              {}, command.Recorder());
		const vk::Extent2D extent {std::max(image.info.extent.width >> view.base_level, 1u),
		                           std::max(image.info.extent.height >> view.base_level, 1u)};
		command.MarkTransfer(CommandBufferDebugOp::InternalWork, address, image.info.data.address,
		                     static_cast<uint32_t>(InternalWork::ColorClearCheck),
		                     static_cast<uint32_t>(slice_size / sizeof(uint32_t)), first + slice);
		ConditionalColorClear::Note note {};
		note.target         = image.info.data.address;
		note.metadata       = address;
		note.layer          = image_first + slice;
		note.width          = extent.width;
		note.height         = extent.height;
		note.samples        = samples;
		note.dcc            = desc.info.metadata.kind == ImageMetadataKind::Dcc;
		note.unwritten      = unwritten;
		note.fg_op          = FrameGraph::Logging() ? FrameGraph::CurrentOp() : 0;
		const auto recorded = m_conditional_clear->Record(
		    command, buffer->BufferDeviceAddress() + offset,
		    static_cast<uint32_t>(slice_size / sizeof(uint32_t)), candidates,
		    image.FindView(target), view.format, samples, extent, note);
		if (check != nullptr && slice == 0) {
			*check = recorded;
		}
		CommitGpuWrite(image);
	}
	static std::atomic<uint64_t> checks {0}, texture_checks {0};
	if (desc.type != BindingType::RenderTarget) {
		texture_checks.fetch_add(count, std::memory_order_relaxed);
	}
	const auto n = checks.fetch_add(count, std::memory_order_relaxed) + count;
	if ((n & (n - 1)) == 0) {
		LOGF("TextureCache: %" PRIu64 " render-target clears checked on the GPU (%" PRIu64
		     " of them for a texture binding)\n",
		     n, texture_checks.load(std::memory_order_relaxed));
	}
	return true;
}

// Diagnostics: GT7's lit pixel shaders loop over a light list whose length they read from a
// two-channel uint 2D-array image (a light grid), and some of their draws take 20-200 ms on the
// GPU, as if the counts were garbage. Which images those are, who binds them and how (a compute
// pass writing one as storage, a pixel shader reading it), and whether the guest bytes behind
// them are GPU- or CPU-written: once per address, for the first few.
// The lit pixel shaders whose draws took 20-200 ms (round 21 dumps), and two of the material
// compute shaders: every image they bind is described, whatever its format. KYTY_DESCRIBE_SHADER_IMAGES=<hash>,<hash>... names others.
static bool DescribesShaderImages(uint64_t shader) {
	static const std::unordered_set<uint64_t> shaders = [] {
		std::unordered_set<uint64_t> result;
		const char* text = std::getenv("KYTY_DESCRIBE_SHADER_IMAGES");
		const std::string list =
		    text != nullptr ? text
		                    : "07c3689281322aba,2a9c7e7ccd1484f9,112704223b775ce7,ed063ee275c1373c,"
		                      "3463527e3cbe7f8c,8a854858e103858d,cb06478e40bc6e99,3ed0b1867a8cbe66,"
		                      "5e4037455eb3c1d8,4c542a4b581e3f2a,542f32f29799fc9d,5a4f390cdcf74502,"
		                      "09f9cc4bc1217575,05c8efa0ef5999ca,6b3182b1b2362382,3ba2ad4871adc373,"
		                      "f0a911d4544b7547,5d8fac8c06b94572,6878a7b8d93bd00e";
		for (size_t at = 0; at < list.size();) {
			const auto end = std::min(list.find(',', at), list.size());
			if (end > at) result.insert(std::strtoull(list.substr(at, end - at).c_str(), nullptr, 16));
			at = end + 1;
		}
		return result;
	}();
	return shader != 0 && shaders.contains(shader);
}

// Diagnostics: the light grid those shaders read (R32G32_UINT, CPU-written): per cell an offset
// into the light-index list (R) and packed light counts (G: bits 0-15, 16-23, 24-31). The guest
// bytes, as the game wrote them, every 64th binding (and the first 8), when a count is large:
// does the game's own data ask for thousands of lights a pixel, or does the emulator?
static void DescribeLightGrid(const ImageInfo& info, uint64_t shader) {
	static std::atomic<uint64_t> bindings {0};
	static std::atomic<uint32_t> lines {0};
	const auto                   n = bindings.fetch_add(1, std::memory_order_relaxed);
	if ((n >= 8 && n % 64 != 0) || lines.load(std::memory_order_relaxed) >= 48) {
		return;
	}
	std::vector<uint32_t> words(static_cast<size_t>(info.data.size / sizeof(uint32_t)));
	if (words.empty() ||
	    !Libs::LibKernel::Memory::TryReadBacking(info.data.address, words.data(),
	                                             words.size() * sizeof(uint32_t))) {
		return;
	}
	const uint32_t row_texels = std::max(info.pitch, info.extent.width);
	const uint64_t layer_size = static_cast<uint64_t>(row_texels) * info.extent.height * 2u;
	uint32_t max_low = 0, max_mid = 0, max_high = 0, max_offset = 0, over = 0, cells = 0;
	for (uint32_t layer = 0; layer < info.resources.layers; layer++) {
		for (uint32_t y = 0; y < info.extent.height; y++) {
			for (uint32_t x = 0; x < info.extent.width; x++) {
				const uint64_t at = layer * layer_size + (static_cast<uint64_t>(y) * row_texels + x) * 2u;
				if (at + 1 >= words.size()) {
					continue;
				}
				const uint32_t offset = words[at];
				const uint32_t packed = words[at + 1];
				cells++;
				max_offset = std::max(max_offset, offset);
				max_low    = std::max(max_low, packed & 0xffffu);
				max_mid    = std::max(max_mid, (packed >> 16u) & 0xffu);
				max_high   = std::max(max_high, packed >> 24u);
				over += (packed & 0xffffu) > 256u ? 1u : 0u;
			}
		}
	}
	if (n >= 8 && max_low <= 256u) {
		return;
	}
	lines.fetch_add(1, std::memory_order_relaxed);
	LOGF("TextureCache: light grid 0x%016" PRIx64 " (%ux%u, %u layers, pitch %u) for shader "
	     "0x%016" PRIx64 ", binding %" PRIu64 ": %u cells, max counts %u / %u / %u, %u cells over "
	     "256 lights, max offset %u\n",
	     info.data.address, info.extent.width, info.extent.height, info.resources.layers,
	     info.pitch, shader, n, cells, max_low, max_mid, max_high, over, max_offset);
}

namespace {

// The shapes pixel shaders bind light grids with (NoteLightGridShape): width, height, row pitch
// and layers, packed under the top bit; 0 is a free slot.
std::array<std::atomic<uint64_t>, 16> g_light_grid_shapes {};

uint64_t LightGridShapeKey(const ImageInfo& info) {
	return (uint64_t {1} << 63u) | (info.extent.width & 0xffffu) |
	       (static_cast<uint64_t>(info.extent.height & 0xffffu) << 16u) |
	       (static_cast<uint64_t>(info.pitch & 0xffffu) << 32u) |
	       (static_cast<uint64_t>(info.resources.layers & 0x7fffu) << 48u);
}

bool LightGridFormat(const ImageInfo& info) {
	return info.pixel_format == vk::Format::eR32G32Uint && !info.IsTiled() && !info.IsVolume() &&
	       info.resources.levels == 1 && info.resources.layers >= 2 && info.samples == 1 &&
	       !info.data.Empty() && info.data.size <= 0x40000u;
}

bool IsLightGridShape(const ImageInfo& info) {
	if (!LightGridFormat(info)) {
		return false;
	}
	const auto key = LightGridShapeKey(info);
	for (const auto& shape: g_light_grid_shapes) {
		const auto value = shape.load(std::memory_order_relaxed);
		if (value == key) {
			return true;
		}
		if (value == 0) {
			return false;
		}
	}
	return false;
}

// No light grid cell of GT7's asks for this many lights (the counts seen are below 64); the
// garbage ones ask for up to 65535.
constexpr uint32_t LightGridMostLights = 1024;

struct LightGridScan {
	uint32_t    cells   = 0;
	uint32_t    garbage = 0;
	uint64_t    first   = UINT64_MAX; // byte offsets of the first and the last such cell
	uint64_t    last    = 0;
	std::string layers; // " <layer>:<cells>" for each layer holding such cells (`describe`)
};

// A light grid's bytes: cell (x, y) of layer l is two words at l * layer bytes + (y * row + x) * 8,
// a light-list offset and packed counts (the low 16 bits one of them). Each cell that asks for
// more than LightGridMostLights lights is emptied in `repair` (`bytes` itself, or a copy), when
// given.
LightGridScan ScanLightGrid(const uint8_t* bytes, uint64_t size, uint32_t width, uint32_t height,
                            uint32_t layers, uint32_t row, uint8_t* repair, bool describe) {
	LightGridScan scan;
	if (layers == 0 || size % layers != 0) {
		return scan;
	}
	const uint64_t layer_bytes = size / layers;
	if (static_cast<uint64_t>(row) * height * 8u > layer_bytes || width > row) {
		return scan;
	}
	for (uint32_t layer = 0; layer < layers; layer++) {
		uint32_t in_layer = 0;
		for (uint32_t y = 0; y < height; y++) {
			for (uint32_t x = 0; x < width; x++) {
				const uint64_t at = layer * layer_bytes + (static_cast<uint64_t>(y) * row + x) * 8u;
				uint32_t       packed = 0;
				std::memcpy(&packed, bytes + at + 4u, sizeof(packed));
				scan.cells++;
				if ((packed & 0xffffu) <= LightGridMostLights) {
					continue;
				}
				in_layer++;
				scan.garbage++;
				scan.first = std::min(scan.first, at);
				scan.last  = std::max(scan.last, at);
				if (repair != nullptr) {
					std::memset(repair + at, 0, 8u);
				}
			}
		}
		if (describe && in_layer != 0) {
			scan.layers += fmt::format(" {}:{}", layer, in_layer);
		}
	}
	return scan;
}

std::string HexWords(const uint8_t* bytes, uint64_t size, uint64_t offset, uint32_t words) {
	std::string text;
	for (uint32_t i = 0; i < words && offset + (i + 1u) * 4u <= size; i++) {
		uint32_t word = 0;
		std::memcpy(&word, bytes + offset + i * 4u, sizeof(word));
		text += fmt::format(" {:08x}", word);
	}
	return text;
}

// A light grid found with such cells, for the background thread (LightGridRechecker): its words
// and the guest memory journal's writes there are logged off the GPU thread, and, for the first
// few, what guest memory holds there is read again 1, 4, 16, 64, 256 and 1000 ms after it was
// found: does the game still write it, or does something write over it later?
struct LightGridFind {
	uint64_t             address  = 0;
	uint32_t             width    = 0;
	uint32_t             height   = 0;
	uint32_t             layers   = 0;
	uint32_t             row      = 0;
	uint64_t             first    = 0; // the first and the last such cell
	uint64_t             last     = 0;
	uint64_t             found_ns = 0; // the journal's clock
	bool                 recheck  = false;
	std::vector<uint8_t> found; // the bytes the check found (before any repair)
	// The rechecks done so far, and their results.
	size_t      step = 0;
	std::string text;
};

class LightGridRechecker final {
public:
	LightGridRechecker(): m_thread([this](const std::stop_token& stop) { Run(stop); }) {}
	~LightGridRechecker() = default;
	KYTY_CLASS_NO_COPY(LightGridRechecker);

	void Queue(LightGridFind find) {
		{
			std::lock_guard lock(m_mutex);
			if (m_queue.size() >= 32) {
				return;
			}
			m_queue.push_back(std::move(find));
		}
		m_ready.notify_one();
	}

private:
	static constexpr std::array<uint32_t, 6> StepsMs {1u, 4u, 16u, 64u, 256u, 1000u};

	static uint64_t Due(const LightGridFind& find) {
		return find.found_ns + uint64_t {StepsMs[find.step]} * 1000000u;
	}

	static void LogDetails(const LightGridFind& find) {
		const auto* bytes     = find.found.data();
		const auto  size      = static_cast<uint64_t>(find.found.size());
		const auto  first_row = Common::AlignDown(find.first, static_cast<uint64_t>(find.row) * 8u);
		LOGF("LightGrid: 0x%012" PRIx64 " words: at +0 (the grid's start):%s | the row of the "
		     "first such cell, +0x%" PRIx64 ":%s | the last such cell, +0x%" PRIx64 ":%s\n",
		     find.address, HexWords(bytes, size, 0, 16).c_str(), first_row,
		     HexWords(bytes, size, first_row, std::min(find.row, 16u) * 2u).c_str(), find.last,
		     HexWords(bytes, size, find.last, 8).c_str());
		const auto age_ms = (LibKernel::Memory::JournalNowNs() - find.found_ns) / 1000000u;
		LOGF("LightGrid: 0x%012" PRIx64 " writes there in the second before it was found (and the "
		     "%" PRIu64 " ms since), newest first: %s\n",
		     find.address, age_ms,
		     LibKernel::Memory::DescribeJournal(find.address, size, 1000u + age_ms + 1u, 40)
		         .c_str());
	}

	// Reads the grid's guest memory once more; true when that was its last recheck.
	static bool Recheck(LightGridFind& find, std::vector<uint8_t>& now) {
		const auto size = static_cast<uint64_t>(find.found.size());
		now.resize(find.found.size());
		const auto age =
		    static_cast<double>(LibKernel::Memory::JournalNowNs() - find.found_ns) / 1e6;
		find.step++;
		if (!LibKernel::Memory::TryReadBacking(find.address, now.data(), size)) {
			find.text += fmt::format(" +{:.1f} ms: unreadable", age);
			return true;
		}
		size_t differ = 0;
		for (uint64_t at = 0; at + 4u <= size; at += 4u) {
			differ += std::memcmp(now.data() + at, find.found.data() + at, 4u) != 0 ? 1u : 0u;
		}
		const auto scan = ScanLightGrid(now.data(), size, find.width, find.height, find.layers,
		                                find.row, nullptr, false);
		find.text += fmt::format(" +{:.1f} ms: {} such cells, {}", age, scan.garbage,
		                         differ == 0 ? std::string("bytes as found")
		                                     : fmt::format("{} words changed", differ));
		return find.step >= StepsMs.size();
	}

	void Run(const std::stop_token& stop) {
		std::vector<LightGridFind> active; // this thread's only
		std::vector<uint8_t>       now;
		std::unique_lock           lock(m_mutex);
		for (;;) {
			while (!m_queue.empty()) {
				auto find = std::move(m_queue.front());
				m_queue.pop_front();
				lock.unlock();
				LogDetails(find);
				if (find.recheck) {
					active.push_back(std::move(find));
				}
				lock.lock();
			}
			if (active.empty()) {
				if (!m_ready.wait(lock, stop, [this] { return !m_queue.empty(); })) {
					return;
				}
				continue;
			}
			// The recheck due first, across the finds.
			size_t next = 0;
			for (size_t i = 1; i < active.size(); i++) {
				if (Due(active[i]) < Due(active[next])) {
					next = i;
				}
			}
			const auto due     = Due(active[next]);
			const auto current = LibKernel::Memory::JournalNowNs();
			if (due > current) {
				(void)m_ready.wait_for(lock, stop, std::chrono::nanoseconds(due - current),
				                       [this] { return !m_queue.empty(); });
				if (stop.stop_requested()) {
					return;
				}
				continue;
			}
			lock.unlock();
			auto& find = active[next];
			if (Recheck(find, now)) {
				const auto since_ms =
				    (LibKernel::Memory::JournalNowNs() - find.found_ns) / 1000000u + 1u;
				LOGF("LightGrid: 0x%012" PRIx64 " in guest memory after it was found:%s; writes "
				     "there since: %s\n",
				     find.address, find.text.c_str(),
				     LibKernel::Memory::DescribeJournal(find.address, find.found.size(), since_ms,
				                                        16)
				         .c_str());
				active.erase(active.begin() + static_cast<std::ptrdiff_t>(next));
			}
			lock.lock();
		}
	}

	std::mutex                  m_mutex;
	std::condition_variable_any m_ready;
	std::deque<LightGridFind>   m_queue;
	std::jthread                m_thread; // last: it starts with the others in place
};

// Made on the first find; stopped (joined) with the texture cache, before guest memory goes.
std::mutex                          g_rechecker_mutex;
std::unique_ptr<LightGridRechecker> g_rechecker;
bool                                g_rechecker_stopped = false;

void QueueLightGridFind(LightGridFind find) {
	std::lock_guard lock(g_rechecker_mutex);
	if (g_rechecker_stopped) {
		return;
	}
	if (g_rechecker == nullptr) {
		g_rechecker = std::make_unique<LightGridRechecker>();
	}
	g_rechecker->Queue(std::move(find));
}

void StopLightGridRechecks() {
	std::unique_ptr<LightGridRechecker> rechecker;
	{
		std::lock_guard lock(g_rechecker_mutex);
		g_rechecker_stopped = true;
		rechecker           = std::move(g_rechecker);
	}
	rechecker.reset();
}

} // namespace

void TextureCache::NoteLightGridShape(const ImageInfo& info) {
	if (!LightGridFormat(info)) {
		return;
	}
	const auto key = LightGridShapeKey(info);
	for (auto& shape: g_light_grid_shapes) {
		auto value = shape.load(std::memory_order_relaxed);
		if (value == key) {
			return;
		}
		if (value == 0) {
			if (shape.compare_exchange_strong(value, key, std::memory_order_relaxed)) {
				LOGF("LightGrid: pixel shaders read light grids of %ux%u cells, %u layers (row "
				     "pitch "
				     "%u): their uploads are checked\n",
				     info.extent.width, info.extent.height, info.resources.layers, info.pitch);
				return;
			}
			if (value == key) {
				return;
			}
		}
	}
}

void TextureCache::CheckLightGridUpload(const Image& image, Buffer& source,
                                        uint64_t source_offset) {
	const auto& info = image.info;
	if (!IsLightGridShape(info)) {
		return;
	}
	const auto range  = info.data;
	const bool staged = &source == &m_buffer_cache.GetUtilityBuffer(MemoryUsage::Upload) &&
	                    source.Mapped().size() >= source_offset &&
	                    source.Mapped().size() - source_offset >= range.size;
	// The guest bytes, read again into cached memory (the staging copy is write-combined, slow to
	// read). When they hold such cells, they are what is uploaded, repaired: the staging copy is
	// overwritten with them, so the cells checked are the cells uploaded even when the game
	// wrote the grid in between. Uploaded from a cached buffer (its bytes GPU-written), the guest
	// bytes are only counted: nothing is repaired.
	thread_local std::vector<uint8_t> guest;
	guest.resize(static_cast<size_t>(range.size));
	if (!LibKernel::Memory::TryReadBacking(range.address, guest.data(), range.size)) {
		return;
	}
	const auto width  = info.extent.width;
	const auto height = info.extent.height;
	const auto layers = info.resources.layers;
	const auto row    = std::max(info.pitch, width);
	const auto scan =
	    ScanLightGrid(guest.data(), range.size, width, height, layers, row, nullptr, true);
	Common::PerfStats::AddLightGrid(scan.garbage);
	if (scan.garbage == 0) {
		return;
	}
	static std::atomic<uint64_t> found {0};
	const auto                   n      = found.fetch_add(1, std::memory_order_relaxed) + 1;
	const bool                   log    = n <= 24 || n % 64 == 0;
	const bool                   repair = staged && !Bisect::LightGridsRaw();
	if (log) {
		const auto writebacks = PendingWriteBacks();
		const auto under = std::count_if(writebacks.begin(), writebacks.end(), [&](const auto& w) {
			return w.first < range.address + range.size && range.address < w.first + w.second;
		});
		LOGF("LightGrid: 0x%012" PRIx64 "+0x%" PRIx64 " (%ux%u, %u layers, pitch %u), %s, from %s: "
		     "%u of %u cells ask for more than %u lights, from +0x%" PRIx64 " to +0x%" PRIx64
		     " (per layer:%s); %s (%" PRIu64 " so far) | GPU-written bytes there: %s, image "
		     "write-backs on their way over it: %zu\n",
		     range.address, range.size, width, height, layers, info.pitch,
		     image.initialized ? "uploaded again" : "a new image",
		     staged ? "its guest bytes" : "a cached buffer (GPU-written)", scan.garbage, scan.cells,
		     LightGridMostLights, scan.first, scan.last, scan.layers.c_str(),
		     repair   ? "uploaded as empty cells"
		     : staged ? "uploaded as they are (grids:raw)"
		              : "uploaded as they are (not repairable)",
		     n, m_buffer_cache.IsRegionGpuModified(range.address, range.size) ? "yes" : "no",
		     static_cast<size_t>(under));
		static std::atomic<uint32_t> rechecks {0};
		LightGridFind                find;
		find.address  = range.address;
		find.width    = width;
		find.height   = height;
		find.layers   = layers;
		find.row      = row;
		find.first    = scan.first;
		find.last     = scan.last;
		find.found_ns = LibKernel::Memory::JournalNowNs();
		find.recheck  = rechecks.fetch_add(1, std::memory_order_relaxed) < 12;
		find.found    = guest;
		QueueLightGridFind(std::move(find));
	}
	if (repair) {
		(void)ScanLightGrid(guest.data(), range.size, width, height, layers, row, guest.data(),
		                    false);
		std::memcpy(source.Mapped().data() + source_offset, guest.data(),
		            static_cast<size_t>(range.size));
		source.Flush(source_offset, range.size);
	}
}

void TextureCache::DescribeUintArrayImage(ImageId id, const ImageDesc& desc) {
	const auto& info   = desc.info;
	const uint64_t shader = BufferCache::s_diag_shader_hash;
	if (desc.type == BindingType::VideoOut || desc.type == BindingType::DepthTarget) {
		return;
	}
	const bool lit_shader = DescribesShaderImages(shader);
	if (lit_shader && info.pixel_format == vk::Format::eR32G32Uint && info.data.size <= 0x100000) {
		DescribeLightGrid(info, shader);
	}
	if (!lit_shader &&
	    (info.IsVolume() || info.resources.layers < 2 || vk::componentCount(info.pixel_format) != 2 ||
	     std::string_view(vk::componentNumericFormat(info.pixel_format, 0)) != "UINT")) {
		return;
	}
	static std::mutex                                   logged_mutex;
	static std::unordered_set<uint64_t>                 addresses;
	static std::unordered_set<uint64_t>                 pairs;
	static uint32_t                                     lit_lines = 0;
	{
		// The first 8 addresses; then each new (shader, binding kind) of them, up to 64. The lit
		// shaders' images: each (image, shader) once, up to 96.
		std::lock_guard lock(logged_mutex);
		if (lit_shader) {
			const auto pair = info.data.address ^ (shader * 0x9e3779b97f4a7c15ull) ^
			                  (static_cast<uint64_t>(desc.type) << 56) ^ 1u;
			if (lit_lines >= 96 || !pairs.insert(pair).second) {
				return;
			}
			lit_lines++;
		} else if (!addresses.contains(info.data.address)) {
			if (addresses.size() >= 8) {
				return;
			}
			addresses.insert(info.data.address);
		}
		if (!lit_shader) {
			const auto pair = info.data.address ^ (shader * 0x9e3779b97f4a7c15ull) ^
			                  (static_cast<uint64_t>(desc.type) << 56);
			if (pairs.size() >= 160 || !pairs.insert(pair).second) {
				return;
			}
		}
	}
	bool gpu_image = false, cpu_dirty = false, buffer_modified = false;
	{
		std::scoped_lock lock {m_lock};
		const auto&      image = m_slot_images[id];
		gpu_image              = image.IsGpuModified();
		cpu_dirty              = image.IsCpuDirty();
		buffer_modified        = image.IsBufferModified();
	}
	const bool gpu_bytes = m_buffer_cache.IsRegionGpuModified(info.data.address, info.data.size);
	const bool cpu_bytes = m_buffer_cache.IsRegionCpuModified(info.data.address, info.data.size);
	LOGF("TextureCache: %s image 0x%016" PRIx64 "+0x%" PRIx64
	     " %ux%u layers %u format %u (%s x%u) guest %u tile %u pitch %u: binding %u by shader 0x%016" PRIx64
	     "; image GPU-written %d, CPU-dirty %d, buffer-modified %d; guest bytes GPU-written %d, "
	     "CPU-written %d\n",
	     lit_shader ? "lit-shader" : "uint2 array", info.data.address, info.data.size,
	     info.extent.width, info.extent.height,
	     info.resources.layers, static_cast<uint32_t>(info.pixel_format),
	     vk::componentCount(info.pixel_format) != 0 ? vk::componentNumericFormat(info.pixel_format, 0)
	                                                : "?",
	     vk::componentCount(info.pixel_format), static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
	     info.pitch, static_cast<uint32_t>(desc.type), shader, gpu_image ? 1 : 0,
	     cpu_dirty ? 1 : 0, buffer_modified ? 1 : 0, gpu_bytes ? 1 : 0, cpu_bytes ? 1 : 0);
	if (vk::componentCount(info.pixel_format) != 0 &&
	    std::string_view(vk::componentNumericFormat(info.pixel_format, 0)) == "UINT") {
		m_buffer_cache.WatchGpuWrites(info.data.address, info.data.size,
		                              lit_shader ? "lit-shader uint image" : "uint2 array image");
	}
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range        = desc.info.metadata.range;
	const auto current_tick = m_scheduler.CurrentTick();
	const auto layers       = desc.info.TransferLayers();
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// A depth target's HTile that starts here too keeps its entry: GT7 reuses memory within a
		// frame, and the HTile of its in-car map's depth target is where an 833x800 dashboard
		// target keeps its CMASK. Erasing the entry here, as before round 64, left that depth
		// target's next HTile clear unseen once round 63b gave multisampled targets their CMASK:
		// its depth kept old values and the map's track failed its depth test (only the cursor
		// was drawn, 2026-10-16). A fill here now clears the depth target and still runs, so this
		// target sees its bytes too (RenderExecutor::TryConsumeComputeMetaClear, IsColorMeta).
		if (m_color_metas.size() >= 4096) {
			m_color_metas.clear();
		}
		m_color_metas.insert(range.address);
		if (const auto shared = m_surface_metas.find(range.address);
		    shared != m_surface_metas.end() && shared->second.type == MetaDataInfo::Type::HTile) {
			static std::unordered_set<uint64_t> logged;
			if (logged.size() < 16 && logged.insert(range.address).second) {
				LOGF("TextureCache: color target 0x%012" PRIx64
				     " %ux%u (%u samples) keeps its %s at "
				     "0x%012" PRIx64 ", where a depth target keeps its HTile: both are kept\n",
				     desc.info.data.address, desc.info.extent.width, desc.info.extent.height,
				     desc.info.samples,
				     desc.info.metadata.kind == ImageMetadataKind::Dcc ? "DCC" : "CMASK",
				     range.address);
			}
		}
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return;
		}
		// A fast-clear stamps the whole DCC metadata range with one uniform code; an ordinary
		// draw also touches these bytes (per-block compression state) but leaves a mixed
		// pattern, so it never passes the all-same-code check below -- yet it re-marks the range
		// GPU-dirty just the same, so gating purely on that flag re-triggers the synchronous
		// readback on every draw to an already-bound render target. Gate on the scheduler tick
		// instead: within one still-unsubmitted recording no GPU write can land in guest memory
		// after the first check, so a later FindImage for the same slices of the same image
		// this tick is redundant. Other slices of the image (another layer or face) keep their
		// own check, a CPU store to the metadata (a guest-side clear) is caught through the
		// tracker below, and a real re-clear in a later recording still gets caught, since the
		// tick will have advanced.
		const bool already_resolved_this_tick =
		    image.dcc_clear_checked_tick == current_tick &&
		    image.dcc_clear_checked_first == first && image.dcc_clear_checked_count == count;
		image.dcc_clear_checked_tick  = current_tick;
		image.dcc_clear_checked_first = first;
		image.dcc_clear_checked_count = count;
		// Round 139: not when something wrote the metadata since that check. A game can fast-clear
		// the same target again in the same recording: a course layout screen draws each map
		// into one multisampled target, its CMASK filled again before each, and a map drawn
		// without a submission between it and the last was drawn over the one before (the check
		// skipped here). The check (or the read of the bytes) then watches the slices, so a
		// target bound again with its metadata untouched still skips it.
		const auto slices = layers != 0 ? range.size / layers : 0;
		if (already_resolved_this_tick &&
		    !m_buffer_cache.IsRegionCpuModified(range.address, range.size) &&
		    (slices == 0 ||
		     m_buffer_cache.MetadataUnwritten(range.address + slices * first, slices * count))) {
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  color clear of 0x%012" PRIx64 " (%s 0x%012" PRIx64
				                ", layers %u+%u): checked already in this recording, nothing wrote "
				                "its metadata since",
				                desc.info.data.address,
				                desc.info.metadata.kind == ImageMetadataKind::Dcc ? "DCC" : "CMASK",
				                range.address, first, count);
			}
			return;
		}
		if (already_resolved_this_tick && FrameGraph::Logging()) {
			FrameGraph::Log("  color clear of 0x%012" PRIx64 " (%s 0x%012" PRIx64
			                ", layers %u+%u): checked again in this recording, its metadata "
			                "written since",
			                desc.info.data.address,
			                desc.info.metadata.kind == ImageMetadataKind::Dcc ? "DCC" : "CMASK",
			                range.address, first, count);
		}
	}
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	// A GPU fill of the metadata (DMA_DATA, GT7's fast clears) leaves a value known here: the
	// download drained the GPU twice a frame in its races (47 % of the GPU thread).
	uint32_t   filled      = 0;
	const bool gpu_written = m_buffer_cache.IsRegionGpuModified(range.address, range.size);
	const bool known_fill =
	    gpu_written && m_buffer_cache.KnownFill(range.address, range.size, filled);
	if (gpu_written) {
		// Logged at powers of two: how often a known fill spared the download.
		static std::atomic<uint64_t> spared {0}, downloaded {0};
		const auto n = (known_fill ? spared : downloaded).fetch_add(1, std::memory_order_relaxed) + 1;
		if ((n & (n - 1)) == 0) {
			LOGF("TextureCache: color metadata of 0x%016" PRIx64 " GPU-written: %" PRIu64
			     " read from a known fill, %" PRIu64 " downloaded\n",
			     range.address, spared.load(std::memory_order_relaxed),
			     downloaded.load(std::memory_order_relaxed));
		}
	}
	// KYTY_DCC_CLEAR_READBACK=0: GPU-written color metadata is taken as not cleared, without the
	// download (an A/B switch: a clear missed this way leaves the target uncleared).
	static const bool readback = [] {
		const char* value = std::getenv("KYTY_DCC_CLEAR_READBACK");
		return value == nullptr || value[0] != '0';
	}();
	// A check on the GPU rewrites the metadata it reads (to expanded keys, when they held one
	// code), so what it found holds until something writes those bytes again. The first draw of
	// each command buffer checked again (one pause of the GPU's work and a render pass break each,
	// 376 a frame in a race); a check that cleared again would also wipe what was drawn before it,
	// a suspect for the course maps' missing track (the outcome of each check is now read back:
	// ConditionalColorClear). KYTY_GPU_CLEAR_RECHECK=1 or KYTY_BISECT's recheck checks again
	// anyway, as before (the check's outcome is then reported as one of unwritten metadata).
	const auto slice_bytes   = range.size / layers;
	const auto checked_begin = range.address + slice_bytes * first;
	const auto checked_bytes = slice_bytes * count;
	if (gpu_written && !known_fill && GpuColorClearEnabled() &&
	    (desc.type != BindingType::VideoOut || !Bisect::FlipCheckOld())) {
		// (A CPU write to bytes the GPU had not written leaves them CPU-dirty, which the watch
		// does not see.)
		const bool unwritten = !m_buffer_cache.IsRegionCpuModified(checked_begin, checked_bytes) &&
		                       m_buffer_cache.MetadataUnwritten(checked_begin, checked_bytes);
		if (unwritten && !RechecksColorClears()) {
			ConditionalColorClear::NoteSkipped();
			if (FrameGraph::Logging()) {
				FrameGraph::Log("  color clear of 0x%012" PRIx64 " (%s 0x%012" PRIx64
				                ", layers %u+%u): nothing wrote its metadata since its last check",
				                desc.info.data.address,
				                desc.info.metadata.kind == ImageMetadataKind::Dcc ? "DCC" : "CMASK",
				                range.address, first, count);
			}
			return;
		}
		uint32_t check = 0;
		if (MaterializeColorClearOnGpu(id, desc, first, image_first, count, slice_bytes, unwritten,
		                               &check)) {
			// After the check's own write of the metadata (ObtainBuffer marked it GPU-written).
			// Only a check that ran: a binding with no code to clear with (a texture's, which
			// cannot decode a register clear) records none, and the render-target binding after
			// it must still check.
			if (check != 0) {
				m_buffer_cache.WatchMetadata(checked_begin, checked_bytes);
			}
			if (FrameGraph::Logging() && check == 0) {
				FrameGraph::Log(
				    "  color clear of 0x%012" PRIx64 " (%s 0x%012" PRIx64
				    ", layers %u+%u): no code this binding could clear with, not checked",
				    desc.info.data.address,
				    desc.info.metadata.kind == ImageMetadataKind::Dcc ? "DCC" : "CMASK",
				    range.address, first, count);
			} else if (FrameGraph::Logging()) {
				FrameGraph::Log("  color clear of 0x%012" PRIx64 " (%s 0x%012" PRIx64
				                ", layers %u+%u, %u samples): checked on the GPU (check %u%s)",
				                desc.info.data.address,
				                desc.info.metadata.kind == ImageMetadataKind::Dcc ? "DCC" : "CMASK",
				                range.address, first, count, desc.info.samples, check,
				                unwritten ? ", its metadata unwritten since the last" : "");
			}
			return;
		}
	}
	if (gpu_written && !known_fill) {
		if (!readback && desc.type == BindingType::RenderTarget) {
			return;
		}
		// Who keeps writing it (the first few ranges): the download drains the GPU each time.
		m_buffer_cache.WatchGpuWrites(range.address, range.size, "color metadata");
		if (desc.type == BindingType::VideoOut) {
			// A flip the check on the GPU declined: between commands, so the bytes are copied on
			// the readback queue if their writer is done (round 80).
			const BufferCache::GpuThreadFastReads fast_reads(m_buffer_cache);
			m_buffer_cache.ReadMemory(range.address, range.size, false);
		} else {
			m_buffer_cache.ReadMemory(range.address, range.size, false);
		}
	}
	static std::atomic<uint64_t> downloads {0}, cleared {0};
	bool                         found_clear = false;
	const auto                   report      = [&] {
		if (!gpu_written || known_fill) return;
		const auto n = downloads.fetch_add(1, std::memory_order_relaxed) + 1;
		if (found_clear) cleared.fetch_add(1, std::memory_order_relaxed);
		if ((n & (n - 1)) == 0) {
			LOGF("TextureCache: color metadata downloads: %" PRIu64 ", a clear found in %" PRIu64
			     "\n",
			     n, cleared.load(std::memory_order_relaxed));
		}
	};
	const auto slice_size = range.size / layers;
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (known_fill) {
			code = static_cast<uint8_t>(filled);
			if (filled != code * 0x01010101u) {
				continue; // not one code throughout
			}
		} else if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			continue;
		}
		std::vector<uint8_t> bytes(slice_size);
		if (!known_fill) {
			if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
				EXIT("TextureCache: failed to read color metadata slice\n");
			}
			if (!std::all_of(bytes.begin(), bytes.end(),
			                 [code](uint8_t byte) { return byte == code; })) {
				continue;
			}
		}
		found_clear = true;
		if (FrameGraph::Logging()) {
			FrameGraph::Log("  color clear of 0x%012" PRIx64 " layer %u (code 0x%02x, %s): cleared",
			                desc.info.data.address, image_first + slice, code,
			                known_fill ? "a known fill" : "its metadata bytes");
		}
		{
			std::scoped_lock lock {m_lock};
			const auto&      cleared = m_slot_images[id];
			const bool       depth   = cleared.info.IsDepth();
			if (depth) {
				const auto value   = clear.color.float32[0];
				clear.depthStencil = vk::ClearDepthStencilValue {value, 0u};
			}
			ClearImage(m_scheduler.Current(), id, depth ? cleared.backing.format : view.format,
			           {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
			            view.base_level, view.level_count, image_first + slice, 1},
			           clear);
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			// Still GPU-written (a known fill): both copies, without the download a CPU store
			// to GPU-written bytes needs.
			const LibKernel::Memory::WriteSourceScope source(
			    LibKernel::Memory::WriteSource::MetadataClear);
			// Round 111: a GPU-thread store, after the deferred memory work.
			m_buffer_cache.FinishDeferredMemoryWork();
			if (!known_fill || !m_buffer_cache.WriteClean(address, bytes.data(), bytes.size())) {
				m_buffer_cache.InvalidateMemory(address, slice_size);
				LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
			}
		}
	}
	// Round 139: what was read here holds until something writes the slices again (after the
	// expanded keys' own write above), so a binding later in this recording skips them.
	m_buffer_cache.WatchMetadata(range.address + slice_size * first, slice_size * count);
	report();
}

// KYTY_GUEST_HASH_IMAGES_KB (default 256, 0: off): linear images up to this size that the GPU
// neither wrote nor covers with its writes are compared with their guest bytes (a hash) the
// first time each guest submission binds them, and uploaded again, straight from guest memory,
// when the bytes changed. What the game wrote before submitting is visible to all of that
// submission's commands, as on the console; a CPU write the page tracking does see in between
// is compared again. The page tracking alone misses CPU writes to an image's first and last pages
// once a CPU write next to the image untracked them (InvalidateCpuAliases), and an upload from a
// buffer the buffer cache keeps over the range is only as fresh as that buffer. GT7's light
// grids, which the CPU rewrites every frame in memory it shares with other data, were found
// stale on the GPU against their guest bytes (frame captures, 2026-10-13), as its tunnels and
// night scenes went dark in screen tiles.
static uint64_t GuestHashImageBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_GUEST_HASH_IMAGES_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull) << 10u;
	}();
	return bytes;
}

// KYTY_GUEST_HASH_TILED_KB (default 0: off): the same check for tiled images up to this size,
// an experiment for textures the CPU streams in (it hashes every one bound, once a submission).
static uint64_t GuestHashTiledBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_GUEST_HASH_TILED_KB");
		const auto  kb    = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		if (kb != 0) {
			LOGF("TextureCache: tiled images up to %llu KB are checked against their guest bytes "
			     "once a submission (KYTY_GUEST_HASH_TILED_KB)\n",
			     static_cast<unsigned long long>(kb));
		}
		return kb << 10u;
	}();
	return bytes;
}

bool TextureCache::ChecksGuestContent(const Image& image) const {
	const auto& info  = image.info;
	const auto  limit = info.IsTiled() ? GuestHashTiledBytes() : GuestHashImageBytes();
	return limit != 0 && !info.data.Empty() && info.data.size <= limit && info.samples == 1 &&
	       !image.depth_id && !info.IsDepth() &&
	       info.metadata.compression == VideoOutCompression::Uncompressed &&
	       !image.IsGpuModified() && !image.IsBufferModified() &&
	       !m_buffer_cache.IsRegionGpuModified(info.data.address, info.data.size);
}

void TextureCache::RefreshByContent(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto range = image.info.data;
	TrackImage(id);
	const auto submission = g_guest_submission_seq.load(std::memory_order_relaxed);
	if (image.content_check_submission == submission && !image.IsCpuDirty()) {
		return;
	}
	uint64_t hash = 0;
	if (!HashGuestBacking(range.address, range.size, hash)) {
		image.content_hash_valid = false;
		RefreshImage(id);
		return;
	}
	image.content_check_submission = submission;
	if (hash == image.content_hash) {
		// The GPU holds these bytes already, whatever the tracking marked.
		if (image.IsDefinitelyCpuDirty()) {
			image.RefreshComplete();
		} else {
			image.ClearMaybeCpuDirty();
		}
		return;
	}
	if (!image.IsDefinitelyCpuDirty()) {
		// What this check is for: the first few, then every 1000th.
		static std::atomic<uint64_t> unseen {0};
		const auto                   count = ++unseen;
		if (count <= 16 || count % 1000 == 0) {
			LOGF("TextureCache: guest bytes of 0x%012" PRIx64 "+0x%" PRIx64
			     " (%ux%u, %u layers, format %u) changed unseen by the page tracking%s: uploaded "
			     "again (%" PRIu64 " so far)\n",
			     range.address, range.size, image.info.extent.width, image.info.extent.height,
			     image.info.resources.layers, static_cast<uint32_t>(image.info.pixel_format),
			     image.IsMaybeCpuDirty() ? " (a neighbour's write marked it maybe-dirty)" : "",
			     count);
		}
	}
	image.MarkCpuDirty();
	InitializeImage(id);
}

void TextureCache::RefreshImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		// Bytes written to such a stencil plane (a buffer write or the CPU) never reach the host's
		// stencil: each plane is logged once (the first 16), as a lead when a stencil-tested draw
		// goes missing. A fill of the whole plane clears it instead (ClearImageFromBuffer).
		if (image.IsBufferModified() || image.IsDefinitelyCpuDirty()) {
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			const auto&                         depth = m_slot_images[image.depth_id].info;
			std::lock_guard                     logged_lock(logged_mutex);
			if (logged.size() < 16 && logged.insert(image.info.data.address).second) {
				LOGF("TextureCache: what %s wrote to the stencil plane 0x%012" PRIx64 "+0x%" PRIx64
				     " of depth target 0x%012" PRIx64 " %ux%u (%u samples%s) does not reach "
				     "its stencil\n",
				     image.IsBufferModified() ? "the GPU" : "the CPU", image.info.data.address,
				     image.info.data.size, depth.data.address, depth.extent.width,
				     depth.extent.height, depth.samples,
				     depth.metadata.stencil_compressed ? ", the stencil in its HTile" : "");
			}
		}
		return;
	}
	if (image.content_hash_valid && ChecksGuestContent(image)) {
		RefreshByContent(id);
		return;
	}
	if (image.info.IsTiled() && ChecksGuestContent(image)) {
		// The KYTY_GUEST_HASH_TILED_KB experiment: an image uploaded before it was checked is
		// uploaded once more from its guest bytes, which records their hash for the next checks.
		TrackImage(id);
		image.MarkCpuDirty();
		InitializeImage(id);
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
	if (record.depth_id != depth_id) {
		if (auto* previous = m_slot_images.try_get(record.depth_id); previous != nullptr) {
			std::erase(previous->stencil_ids, association);
		}
		depth.stencil_ids.push_back(association);
	}
	record.depth_id             = depth_id;
	record.stencil_subresources = depth.stencil_subresources;
	return association;
}

namespace {

// A binding the overlap loop found inside an image that starts elsewhere, where it has no place
// its view fits in (TextureCache::FindImage): it gets an image of its own. Logged, the first 16
// and then at each power of two.
void NoteUnplacedBinding(const TextureCache::ImageDesc& desc, const ImageInfo& found, int32_t mip,
                         int32_t layer) {
	static std::atomic<uint64_t> unplaced {0};
	const auto                   count = unplaced.fetch_add(1, std::memory_order_relaxed) + 1;
	if (count > 16u && (count & (count - 1u)) != 0u) {
		return;
	}
	LOGF("TextureCache: 0x%012" PRIx64 " (%ux%u, mips %u, layers %u, view mips %u+%u layers %u+%u, "
	     "format %u) lies in 0x%012" PRIx64
	     " (%ux%u, mips %u, layers %u) %s: given an image of its "
	     "own (%" PRIu64 " so far)\n",
	     desc.info.data.address, desc.info.extent.width, desc.info.extent.height,
	     desc.info.resources.levels, desc.info.resources.layers, desc.view_info.base_level,
	     desc.view_info.level_count, desc.view_info.base_layer, desc.view_info.layer_count,
	     static_cast<uint32_t>(desc.info.pixel_format), found.data.address, found.extent.width,
	     found.extent.height, found.resources.levels, found.resources.layers,
	     mip < 0 || layer < 0
	         ? "at no mip and layer"
	         : fmt::format("at mip {} layer {}, where its view does not fit", mip, layer).c_str(),
	     count);
}

} // namespace

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format, bool* same_backing) {
	if (same_backing != nullptr) {
		*same_backing = false;
	}
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
		// An image with the same backing starts at the same address, so it is among the owners of
		// the first page: those are looked at first, and the overlaps in the rest of the range only
		// when none of them is. The images of a binding found again (nearly all of them) were
		// otherwise looked up page by page over their whole size, a page table lookup each 16 KB
		// (3 % of the GPU thread in a race, round 68).
		ImageIds candidates;
		// alias:old (an A/B step): the read-only aliases are not found as they are; the overlap
		// loop replaces them as before round 79.
		const bool hide_aliases = m_read_only_aliases != 0 && Bisect::ReadOnlyAliasesOff();
		if (const auto* owners =
		        m_image_page_table.Find(desc.info.data.address >> ImagePageTable::kPageBits);
		    owners != nullptr) {
			owners->ForEach([&](ImageId id) {
				const auto* image = m_slot_images.try_get(id);
				if (image != nullptr && SameBacking(image->info, desc.info, exact_format) &&
				    !(hide_aliases && image->read_only_alias)) {
					result = id;
				}
			});
		}
		if (!result) {
			candidates = FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
			for (const auto id: candidates) {
				const auto& image = m_slot_images[id];
				if (SameBacking(image.info, desc.info, exact_format) &&
				    !(hide_aliases && image.read_only_alias)) {
					result = id;
				}
			}
		}
		// Found with the same backing: the overlap loop below does not run.
		const bool backing_match = static_cast<bool>(result);
		// The images the overlap loop left beside the one this binding gets (round 79).
		ImageIds kept_aliases;

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			// Where the binding lies in the image the loop ends with. Each overlap places the
			// image it was given (the binding, or the image merged so far) in the candidate it
			// returns; before round 74 the last overlap's place was used as the binding's, and a
			// later candidate that returned the image found without a place (or the place of the
			// merged image instead of the binding's) left the binding viewed from its own first
			// layer of an image that starts elsewhere: GT7's reflection faces read their light
			// grids layers before their T#s, in a 48-layer grid of an earlier frame (the light
			// counts of 65535). KYTY_BISECT's views:old places it as before.
			const bool old_views  = Bisect::OldViewBase();
			ImageId    placed_in  = {};
			bool       placed_raw = false; // placed against the binding's own address
			for (const auto candidate: candidates) {
				// Freed by an earlier overlap of this loop (a GPU write there frees read-only
				// aliases, round 79): nothing to resolve against.
				if (const auto* live = m_slot_images.try_get(candidate);
				    live == nullptr || !live->registered) {
					continue;
				}
				if (old_views) {
					view_mip   = -1;
					view_layer = -1;
				}
				// The overlap places the binding itself when it is given the binding, or an image
				// that starts where the binding does (an alias of another format, a depth image).
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const bool  direct      = merged_info.data.address == desc.info.data.address;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.kept_alias) {
					kept_aliases.push_back(candidate);
				}
				if (overlap.image) {
					result = overlap.image;
					if (old_views || overlap.mip >= 0 || overlap.layer >= 0) {
						view_mip   = overlap.mip;
						view_layer = overlap.layer;
					}
					placed_in = overlap.mip >= 0 || overlap.layer >= 0 ? overlap.image : ImageId {};
					placed_raw = direct;
				}
			}
			if (result && !old_views) {
				const auto& found      = m_slot_images[result].info;
				const auto  loop_mip   = view_mip;
				const auto  loop_layer = view_layer;
				const bool  loop_own   = placed_in == result && placed_raw;
				view_mip               = -1;
				view_layer             = -1;
				if (found.data.address != desc.info.data.address) {
					// The binding's own place in the image found, else the overlap's if it placed
					// the binding itself there. With neither, or a place its view does not fit
					// in, the binding gets an image of its own instead of a view that starts
					// elsewhere.
					if (const auto mip = desc.info.MipOf(found); mip >= 0) {
						view_mip   = mip;
						view_layer = desc.info.SliceOf(found, mip);
					} else if (loop_own) {
						view_mip   = loop_mip;
						view_layer = loop_layer;
					}
					const bool fits = view_mip >= 0 && view_layer >= 0 &&
					                  static_cast<uint64_t>(view_mip) + desc.view_info.base_level +
					                          desc.view_info.level_count <=
					                      found.resources.levels &&
					                  static_cast<uint64_t>(view_layer) +
					                          desc.view_info.base_layer +
					                          desc.view_info.layer_count <=
					                      found.resources.layers;
					if (!fits) {
						NoteUnplacedBinding(desc, found, view_mip, view_layer);
						view_mip   = -1;
						view_layer = -1;
						result     = {};
					}
				} else if (loop_own) {
					view_mip   = loop_mip;
					view_layer = loop_layer;
				}
			}
		}

		if (result) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result, "a binding needs more mips or layers than it has");
				result = {};
			}
		}
		if (same_backing != nullptr) {
			*same_backing = backing_match && static_cast<bool>(result);
		}
		if (!result) {
			result         = InsertImage(desc.info);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
		}
		if (!kept_aliases.empty()) {
			// The images the overlap loop kept beside the one this binding gets are aliases of
			// it, made or found: flagged together, so a GPU write to any of them frees the
			// others. When it is not one a read-only alias may be (written by the GPU, or
			// elsewhere), the kept ones go, as before round 79.
			auto&      got = m_slot_images[result];
			const bool pair =
			    ReadOnlyAliasCandidate(got) && got.info.data.address == desc.info.data.address;
			kept_aliases.ForEach([&](ImageId id) {
				auto* kept = m_slot_images.try_get(id);
				if (kept == nullptr || !kept->registered || id == result) {
					return;
				}
				if (pair) {
					MarkReadOnlyAlias(*kept);
					return;
				}
				if (kept->binding.is_bound || kept->binding.is_target) {
					kept->binding.needs_rebind = true;
				}
				FreeImage(id, "an image of another kind over the same memory");
			});
			if (pair) {
				MarkReadOnlyAlias(got);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
					LOGF("TextureCache: 0x%012" PRIx64 "+0x%" PRIx64
					     " %ux%ux%u fmt %d beside %zu image(s) of another kind there, kept as "
					     "read-only aliases (round 79; %" PRIu64 " now)\n",
					     got.info.data.address, got.info.data.size, got.info.extent.width,
					     got.info.extent.height, got.info.extent.depth,
					     static_cast<int>(got.info.pixel_format), kept_aliases.size(),
					     m_read_only_aliases);
				}
			}
		}
		auto& image = m_slot_images[result];
		// A binding found inside a larger image starts at its mip view_mip, layer view_layer (where
		// the binding's memory starts); the binding's view starts there plus its own first mip and
		// layer (the T#'s BASE_LEVEL and BASE_ARRAY, a target's SLICE_START). Before round 71 its
		// own were dropped: GT7's reflection cube faces read their light grids through T#s of one
		// 48-layer grid with BASE_ARRAY 0, 8, ... 40, and a face whose grid lay inside an older,
		// larger image read the first face's layers, or other data there (light counts of 65535 a
		// pixel). KYTY_BISECT's views:old drops them again.
		const bool old_views = Bisect::OldViewBase();
		if (view_mip >= 0) {
			const auto own   = desc.view_info.base_level;
			auto       level = static_cast<uint32_t>(view_mip) + (old_views ? 0u : own);
			if (level + desc.view_info.level_count > image.info.resources.levels) {
				level = static_cast<uint32_t>(view_mip);
			}
			desc.view_info.base_level = level;
		}
		if (view_layer >= 0) {
			const auto own   = desc.view_info.base_layer;
			auto       layer = static_cast<uint32_t>(view_layer) + (old_views ? 0u : own);
			const bool fits  = layer + desc.view_info.layer_count <= image.info.resources.layers;
			if (!fits) {
				layer = static_cast<uint32_t>(view_layer);
			}
			if (own != 0 && !old_views) {
				static std::atomic<uint64_t> moved {0};
				const auto                   count = ++moved;
				if (count <= 12u || (count & (count - 1u)) == 0u) {
					LOGF("TextureCache: 0x%012" PRIx64
					     " (%ux%u, layers %u, view %u+%u) lies at layer "
					     "%d of 0x%012" PRIx64 " (%u layers): viewed from layer %u%s (%" PRIu64
					     " so far; before round 71 from layer %d)\n",
					     desc.info.data.address, desc.info.extent.width, desc.info.extent.height,
					     desc.info.resources.layers, own, desc.view_info.layer_count, view_layer,
					     image.info.data.address, image.info.resources.layers, layer,
					     fits ? "" : " (its own first layer does not fit: left out)", count,
					     view_layer);
				}
			}
			desc.view_info.base_layer = layer;
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
	}
	DescribeUintArrayImage(result, desc);
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

bool TextureCache::RefindTarget(ImageId id, const ImageDesc& desc) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	{
		std::scoped_lock lock {m_lock};
		auto*            image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered) {
			return false;
		}
		image->tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(*image);
	}
	// As FindImage after its lookup; a same-backing find leaves the view where the description
	// put it, so its first layer is the metadata's.
	DescribeUintArrayImage(id, desc);
	MaterializeColorClear(id, desc, desc.view_info.base_layer);
	return true;
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

bool TextureCache::ReusableTextureImage(const Image& image) const {
	const auto& info = image.info;
	if (!image.registered || info.data.Empty() || image.depth_id || info.IsDepth() ||
	    info.HasStencil() || info.HasMetadata() ||
	    info.metadata.compression != VideoOutCompression::Uncompressed ||
	    image.binding.needs_rebind) {
		return false;
	}
	// RefreshImage compares such an image's bytes with the guest's once a guest submission.
	const auto limit = info.IsTiled() ? GuestHashTiledBytes() : GuestHashImageBytes();
	return limit == 0 || info.data.size > limit;
}

bool TextureCache::MayReuseTextureBinding(ImageId id) {
	std::scoped_lock lock {m_lock};
	const auto*      image = m_slot_images.try_get(id);
	return image != nullptr && ReusableTextureImage(*image);
}

bool TextureCache::ReuseTextureBinding(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(id);
	if (image == nullptr || !ReusableTextureImage(*image) || image->IsCpuDirty() ||
	    image->IsBufferModified()) {
		return false;
	}
	// TrackImage would track it again (a CPU write next to it untracked part of it).
	if (image->track_addr != image->info.data.address ||
	    image->track_addr_end != image->info.data.End()) {
		return false;
	}
	// As FindImage: its last use is in the command buffer being recorded.
	image->tick_accessed_last = m_scheduler.CurrentTick();
	TouchImage(*image);
	return true;
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
	if (TracesImage(id)) {
		TraceTargetBound(id, desc, "render");
	}
	image.MarkGpuModified();
	image.usage.render_target = true;
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
	if (TracesImage(id)) {
		TraceTargetBound(id, desc, "depth");
	}
	image.MarkGpuModified();
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
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		const auto [meta, inserted] =
		    m_surface_metas.emplace(desc.info.metadata.range.address,
		                            MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                                          .clear_mask = image.info.htile_clear_mask});
		(void)inserted;
		// The largest binding's: a binding of fewer slices has a smaller HTile range.
		meta->second.size = std::max(meta->second.size, desc.info.metadata.range.size);
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
	// The GPU's pixels from now on, not the guest bytes an upload took.
	image.content_hash_valid = false;
	if (m_read_only_aliases != 0) {
		FreeReadOnlyAliasesOf(image);
	}
	// Aliases aged by frames stay alive side by side. A single owner among them: this write
	// supersedes the bytes of every other live image overlapping it, so none of those may later
	// be written back (collector, linear readback, buffer sync) over it. Their native contents
	// stay as they are; the tick rule freed them without a write-back. Scanned only when the
	// owner changes, not on every bind.
	if (AliasAgeByFrames() && !image.alias_owner && !image.depth_id && image.registered &&
	    !image.info.data.Empty()) {
		for (const auto id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
			auto* other = m_slot_images.try_get(id);
			if (other == nullptr || other == &image || other->depth_id || !other->registered) {
				continue;
			}
			other->alias_owner        = false;
			other->content_hash_valid = false;
			if (other->IsGpuModified()) {
				other->ClearGpuModified();
			}
		}
		image.alias_owner = true;
	}
	image.MarkGpuModified();
	image.gpu_pixels = true;
	image.content_serial++;
	if (ClearsTargetsFromFills() && !image.depth_id && !image.info.data.Empty()) {
		// Its pixels are newer than a fill of its memory: an image made there later must not
		// start as that fill (ClearFromKnownFill).
		m_buffer_cache.ForgetFillsUnder(image.info.data.address, image.info.data.size);
	}
	if (m_nan_check != nullptr) {
		NoteNanCheckCandidate(image);
	}
	if (m_trace != nullptr) {
		TraceImageChanged(image, "draw, clear or copy", true);
	}
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear, const BufferFillClear& how) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	const auto refuse = [&how](std::string why) {
		if (how.what != nullptr) {
			*how.what = std::move(why);
		}
		return false;
	};
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	const auto            found = FindImagesInRegion(address, size, false);
	for (const auto id: found) {
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
			return refuse("two images match it");
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return refuse("no image matches it");
	}
	// A stencil plane no buffer write reaches (TextureCache::RefreshImage): multisampled, or kept
	// in the depth target's HTile.
	const auto unreachable_plane = [this](const Image& plane) {
		const auto* depth = plane.depth_id ? m_slot_images.try_get(plane.depth_id) : nullptr;
		return depth != nullptr &&
		       (depth->info.samples != 1 || depth->info.metadata.stencil_compressed);
	};
	bool run_too = false;
	if (how.alone) {
		for (const auto id: found) {
			const auto* other = m_slot_images.try_get(id);
			if (other == nullptr || id == selected || id == stencil_id ||
			    other->info.data.address != address || other->depth_id == selected ||
			    unreachable_plane(*other)) {
				continue;
			}
			// The selected plane is out of a buffer write's reach: clear it, and let the fill
			// still run for the other image.
			if (stencil_id && how.also_run != nullptr &&
			    unreachable_plane(m_slot_images[stencil_id])) {
				run_too = true;
				continue;
			}
			return refuse(fmt::format("image 0x{:012x}+0x{:x} {}x{} format {} starts there too",
			                          other->info.data.address, other->info.data.size,
			                          other->info.extent.width, other->info.extent.height,
			                          static_cast<uint32_t>(other->info.pixel_format)));
		}
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeFillWord(image.info.pixel_format,
		                    GuestColorWordToHost(packed_clear, image.info.GetColorTransform()),
		                    clear.color)) {
			return refuse("its word does not decode for the image's format");
		}
	} else {
		uint8_t stencil_clear = 0;
		bool    decoded       = true;
		if (aspect == vk::ImageAspectFlagBits::eDepth) {
			if (how.guest_word && image.info.bytes_per_block == 2) {
				// Two 16-bit texels a word: both the same.
				decoded                  = (packed_clear >> 16u) == (packed_clear & 0xffffu);
				clear.depthStencil.depth = static_cast<float>(packed_clear & 0xffffu) / 65535.0f;
			} else {
				decoded = DecodePackedDepthClear(image.info.pixel_format, packed_clear,
				                                 clear.depthStencil.depth);
			}
		} else {
			decoded = DecodePackedStencilClear(packed_clear, stencil_clear);
		}
		if (!decoded) {
			return refuse("its word does not decode for the image's format");
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
	if (how.also_run != nullptr) {
		*how.also_run = run_too;
	}
	if (how.what != nullptr) {
		const auto& info = image.info;
		*how.what = fmt::format("the {} of {} 0x{:012x} {}x{}, {} layers, {} samples, format {}{}",
		                        aspect == vk::ImageAspectFlagBits::eStencil ? "stencil"
		                        : aspect == vk::ImageAspectFlagBits::eDepth ? "depth"
		                                                                    : "color",
		                        info.IsDepth() ? "depth target" : "image", info.data.address,
		                        info.extent.width, info.extent.height, info.resources.layers,
		                        info.samples, static_cast<uint32_t>(info.pixel_format),
		                        run_too ? " (and runs, for another image there)" : "");
	}
	return true;
}

bool TextureCache::ClearsTargetsFromFills() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PENDING_FILLS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool TextureCache::ClearFromKnownFill(ImageId id) {
	auto&       image = m_slot_images[id];
	const auto& info  = image.info;
	// A target the draw or dispatch binding it is about to write, made from scratch: a texture
	// may hold what the CPU wrote into memory a fill's holes left unmapped (those stores are not
	// seen), a target's bytes are the GPU's.
	if (!ClearsTargetsFromFills() || image.depth_id || image.backing.image == nullptr ||
	    info.samples != 1 || image.backing.samples != 1 || info.data.Empty() || info.IsVolume() ||
	    info.HasStencil() || !(image.usage.render_target || image.usage.depth_target) ||
	    image.usage.texture || image.usage.storage || image.usage.video_out ||
	    info.metadata.compression != VideoOutCompression::Uncompressed) {
		return false;
	}
	uint32_t value = 0;
	if (!m_buffer_cache.FillOfNewImage(info.data.address, info.data.size, value, false)) {
		return false;
	}
	vk::ClearValue       clear {};
	vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
	if (info.IsDepth()) {
		aspect = ImageViewOps::DepthAspectMask(image.backing.format);
		if (info.pixel_format == vk::Format::eD16Unorm) {
			// Two texels a dword: both the same.
			if ((value >> 16u) != (value & 0xffffu)) {
				return false;
			}
			clear.depthStencil.depth = static_cast<float>(value & 0xffffu) / 65535.0f;
		} else if (!DecodePackedDepthClear(info.pixel_format, value, clear.depthStencil.depth)) {
			return false;
		}
		clear.depthStencil.stencil = 0;
	} else if (!DecodeFillWord(info.pixel_format,
	                           GuestColorWordToHost(value, info.GetColorTransform()),
	                           clear.color)) {
		return false;
	}
	if (!m_buffer_cache.FillOfNewImage(info.data.address, info.data.size, value, true)) {
		return false;
	}
	ClearImage(m_scheduler.Current(), id, image.backing.format,
	           {aspect, 0, info.resources.levels, 0, image.backing.layers}, clear);
	static std::atomic<uint64_t> cleared {0};
	const auto                   count = ++cleared;
	if (count <= 32 || std::has_single_bit(count)) {
		LOGF("TextureCache: %s 0x%012" PRIx64 "+0x%" PRIx64 " %ux%u layers %u mips %u fmt %d "
		     "tile %u created inside a GPU fill with 0x%08x: starts as the fill, not as its memory "
		     "(%" PRIu64 " so far)\n",
		     info.IsDepth() ? "depth target" : "render target", info.data.address, info.data.size,
		     info.extent.width, info.extent.height, info.resources.layers, info.resources.levels,
		     static_cast<int>(info.pixel_format), static_cast<uint32_t>(info.tile_mode), value,
		     count);
	}
	if (TracesImage(id)) {
		TraceImageEvent(
		    id, fmt::format("cleared to the GPU fill 0x{:08x} it was created in", value), true);
	}
	return true;
}

void TextureCache::ZeroNewImage(ImageId id) {
	auto& image   = m_slot_images[id];
	auto& command = m_scheduler.Current();
	if (image.depth_id || image.backing.image == nullptr || command.IsInvalid()) {
		return;
	}
	const auto&                     info  = image.info;
	const bool                      depth = info.IsDepth();
	const vk::ImageSubresourceRange range {
	    depth ? ImageViewOps::DepthAspectMask(image.backing.format)
	          : vk::ImageAspectFlagBits::eColor,
	    0, info.resources.levels, 0, info.IsVolume() ? 1u : image.backing.layers};
	command.EndRendering();
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Recorder());
	if (depth) {
		const vk::ClearDepthStencilValue zero {0.0f, 0};
		command.Recorder().clearDepthStencilImage(
		    image.backing.image, vk::ImageLayout::eTransferDstOptimal, &zero, 1, &range);
	} else {
		const vk::ClearColorValue zero {};
		command.Recorder().clearColorImage(image.backing.image,
		                                   vk::ImageLayout::eTransferDstOptimal, &zero, 1, &range);
	}
	static std::atomic<uint64_t> cleared {0};
	const auto                   count = ++cleared;
	if (count <= 32 || std::has_single_bit(count)) {
		LOGF("TextureCache: new %s 0x%012" PRIx64 "+0x%" PRIx64 " %ux%u layers %u mips %u "
		     "samples %u fmt %d%s cleared to zero (KYTY_ZERO_NEW_MEMORY; %" PRIu64 " so far)\n",
		     depth ? "depth image" : "image", info.data.address, info.data.size, info.extent.width,
		     info.extent.height, info.resources.layers, info.resources.levels, info.samples,
		     static_cast<int>(info.pixel_format),
		     info.metadata.compression != VideoOutCompression::Uncompressed ? " (compressed)" : "",
		     count);
	}
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
	command.MarkTransfer(CommandBufferDebugOp::ImageClear, image.info.data.address, 0,
	                     range.baseMipLevel | (range.levelCount << 16u),
	                     range.baseArrayLayer | (range.layerCount << 16u),
	                     static_cast<uint32_t>(format));
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
		command.Recorder().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Recorder().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	// Images register their pages before they are tracked: a CPU write where none is has nothing
	// to invalidate. Most write faults land there, and need not wait for the lock.
	if (!MayHaveImages(address, size)) {
		return;
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
}

bool TextureCache::InvalidateMemoryNear(uint64_t address, uint64_t size, uint64_t block,
                                        uint64_t block_size, uint64_t* image_pages) {
	if (!GuestRange {address, size}.Valid() || !GuestRange {block, block_size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	if (image_pages != nullptr) {
		*image_pages = 0;
	}
	if (!MayHaveImages(std::min(address, block),
	                   std::max(address + size, block + block_size) - std::min(address, block))) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
	const auto found = FindImagesInRegion(block, block_size, true);
	if (found.empty()) {
		return false;
	}
	if (image_pages != nullptr) {
		// The pages the texture cache watches for each image (its tracked range: an image a CPU
		// write untracked is uploaded again at its next use, and its pages need no fault of
		// their own until then).
		const uint64_t pages = block_size / TRACKER_PAGE_SIZE;
		if (pages > 64 || block_size % TRACKER_PAGE_SIZE != 0 || block % TRACKER_PAGE_SIZE != 0) {
			*image_pages = ~uint64_t {0};
			return true;
		}
		const auto block_end = block + block_size;
		uint64_t   mask      = 0;
		for (const auto id: found) {
			const auto* image = m_slot_images.try_get(id);
			if (image == nullptr || image->info.data.Empty()) {
				mask = pages == 64 ? ~uint64_t {0} : (uint64_t {1} << pages) - 1u;
				break;
			}
			if (!image->IsTracked()) {
				continue;
			}
			const auto begin =
			    std::max(Common::AlignDown(image->track_addr, TRACKER_PAGE_SIZE), block);
			const auto end =
			    std::min(Common::AlignUp(image->track_addr_end, TRACKER_PAGE_SIZE), block_end);
			for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
				mask |= uint64_t {1} << ((page - block) / TRACKER_PAGE_SIZE);
			}
		}
		*image_pages = mask;
	}
	return true;
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
	if (guest_slice > full_slice_size) {
		// Round 123: as in UploadImage, a slice its range cannot hold is not written back.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("TextureCache: depth image 0x%012" PRIx64 "+0x%" PRIx64
			     " %ux%u, %u layers: a slice needs 0x%" PRIx64 " bytes and has 0x%" PRIx64
			     "; not written back (round 123)\n",
			     info.data.address, info.data.size, info.extent.width, info.extent.height, layers,
			     guest_slice, full_slice_size);
		}
		return;
	}
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
	const auto transform = texture.color_transform;
	if (texture.tiles.empty()) {
		if (transform != ColorTransform::None) {
			auto linear = m_tiler.GetScratchBuffer(destination_size);
			image.Download(texture.regions, linear.buffer, 0, linear.size);
			m_tiler.TransformColor(linear,
			                       {destination.Handle(), destination_offset, destination_size},
			                       transform, false);
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
	{
		// A GPU write to the buffer too: a fast readback of these bytes waits for this tick.
		std::unique_lock dirty_lock(m_dirty_ranges_mutex);
		m_write_ticks.Assign(image.info.data.address, image.info.data.address + copy_size,
		                     m_scheduler.CurrentTick());
		NoteMetadataWrite(image.info.data.address, copy_size);
	}
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id, bool evicting,
                                       std::shared_ptr<std::atomic<bool>>* landed) {
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
	// An image larger than the download ring gets a staging buffer of its own for this one
	// transfer: the ring cannot map it, and its regions cannot be cut into ring-sized pieces
	// the way a buffer download can.
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
	if (TracesImage(id)) {
		TraceImageEvent(id,
		                evicting ? "written back to guest memory (evicted)"
		                         : "written back to guest memory",
		                false);
	}
	// Guest memory holds its pixels once this lands (unless the guest writes them first): freeing
	// or reloading it loses nothing the GPU wrote.
	image.gpu_pixels = false;
	const bool guard = evicting && EvictWriteBackGuard();
	// A small read-back for the CPU lands after the guest may already have seen the GPU finish
	// (labels are written as they are recorded): a CPU that read the bytes and then changed them
	// (cleared them, say) keeps its change. Bytes an earlier read-back left are not the guest's.
	// A CPU write that leaves the bytes as they were goes unnoticed.
	const bool readback_guard = !evicting && range.size <= SmallReadbackBytes();
	// The hash of the guest bytes this copy holds (not a second read of guest memory).
	const uint64_t guest_hash = guard || readback_guard ? HashBytes(mapped, range.size) : 0;
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
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	std::shared_ptr<std::atomic<bool>> landed_flag;
	if (landed != nullptr) {
		landed_flag = std::make_shared<std::atomic<bool>>(false);
		*landed     = landed_flag;
	}
	// Until it lands, a fast readback leaves the bytes under it (PendingWriteBacks).
	auto pending = m_pending_writebacks;
	{
		std::lock_guard lock(pending->mutex);
		pending->ranges.emplace_back(range.address, range.size);
	}
	m_scheduler.DeferPriorityOperation([download, dedicated, range, mapped, offset, guard,
	                                    readback_guard, guest_hash, landed_flag, pending, evicting,
	                                    buffer_cache = &m_buffer_cache] {
		struct Landed {
			const std::shared_ptr<std::atomic<bool>>& flag;
			~Landed() {
				if (flag != nullptr) flag->store(true, std::memory_order_release);
			}
		} landed_scope {landed_flag};
		// Last (declared after the flag's scope): landed, dropped or not, it is no longer pending.
		struct Pending {
			const std::shared_ptr<PendingWriteBackList>& list;
			std::pair<uint64_t, uint64_t>                entry;
			~Pending() {
				std::lock_guard lock(list->mutex);
				auto&           ranges = list->ranges;
				if (const auto found = std::find(ranges.begin(), ranges.end(), entry);
				    found != ranges.end()) {
					*found = ranges.back();
					ranges.pop_back();
				}
			}
		} pending_scope {pending, {range.address, range.size}};
		download->Invalidate(offset, range.size);
		uint64_t current = 0;
		if (guard && (!HashGuestBacking(range.address, range.size, current) ||
		              current != guest_hash)) {
			// The guest wrote (or unmapped) these bytes after the image was evicted: they are
			// newer than its pixels.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("TextureCache: evicted image write-back dropped, guest=0x%016" PRIx64
				     " size=0x%" PRIx64 ": the guest bytes changed before it landed\n",
				     range.address, range.size);
			}
			return;
		}
		if (readback_guard &&
		    (!HashGuestBacking(range.address, range.size, current) ||
		     (current != guest_hash && current != LandedReadbackHash(range.address)))) {
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("TextureCache: read-back for the CPU dropped, guest=0x%016" PRIx64
				     " size=0x%" PRIx64 ": the guest wrote the bytes after it was recorded\n",
				     range.address, range.size);
			}
			NoteReadbackLanded(range.address, mapped, range.size, true);
			return;
		}
		const LibKernel::Memory::WriteSourceScope source(
		    LibKernel::Memory::WriteSource::ImageWriteBack);
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
		// Round 140 (from Franseven): the backing write takes no write fault, so cached buffers
		// over these bytes (a pool the image lives in) still hold the old ones while the tracker
		// holds them clean (grass vertices wrong after a race, then a time trial). Evicted images
		// only: a kept image's pages turned CPU-dirty would upload this copy over GPU pixels
		// drawn since it was recorded.
		if (evicting && buffer_cache->RedirtyMapped(range.address, range.size) != 0) {
			buffer_cache->MarkMappedPending(range.address, range.size);
		}
		if (readback_guard) {
			NoteLandedReadback(range.address, mapped, range.size);
			NoteReadbackLanded(range.address, mapped, range.size, false);
		}
	});
	return true;
}

// A light grid's cells (two words each: a light-list offset and packed light counts) with any
// light, and the most lights a cell holds, per layer: " <cells>/<most>" each.
static std::string LitCellsPerLayer(const uint32_t* cells, size_t words, uint32_t width,
                                    uint32_t height, uint32_t layers, uint32_t pitch) {
	const uint64_t layer_words = words / std::max(layers, 1u);
	std::string    text;
	for (uint32_t layer = 0; layer < layers; layer++) {
		uint32_t count = 0, most = 0;
		for (uint32_t y = 0; y < height; y++) {
			for (uint32_t x = 0; x < width; x++) {
				const uint64_t at =
				    layer * layer_words + (static_cast<uint64_t>(y) * pitch + x) * 2u + 1u;
				if (at >= words) {
					continue;
				}
				count += cells[at] != 0 ? 1u : 0u;
				most = std::max(most, cells[at] & 0xffffu);
			}
		}
		text += fmt::format(" {}/{}", count, most);
	}
	return text;
}

void TextureCache::CompareWithGuest(ImageId id, uint32_t op) {
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(id);
	if (image == nullptr || image->depth_id || image->info.data.Empty()) {
		return;
	}
	const auto range = image->info.data;
	// As the draw's binding will see it: a stage is logged before its textures are bound
	// (RebindImages), so the guest bytes the tracking already marked are uploaded first. The
	// round 40 captures compared the image before that upload.
	const bool tracked_dirty = image->IsCpuDirty();
	RefreshImage(id);
	const auto flags =
	    std::string {image->IsGpuModified() ? 'G' : '-', image->IsBufferModified() ? 'B' : '-',
	                 tracked_dirty ? 'C' : '-'};
	auto transfer = BuildDownload(*image);
	if (!transfer.valid || transfer.depth_target) {
		LOGF("FG %u grid 0x%012" PRIx64 "+0x%" PRIx64 " [%s]: no download for it\n", op,
		     range.address, range.size, flags.c_str());
		return;
	}
	auto guest = std::make_shared<std::vector<uint32_t>>(range.size / sizeof(uint32_t));
	if (guest->empty() || !LibKernel::Memory::TryReadBacking(range.address, guest->data(),
	                                                         guest->size() * sizeof(uint32_t))) {
		return;
	}
	auto download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         AllFlags, range.size);
	if (download->Mapped().data() == nullptr) {
		return;
	}
	// The guest bytes first: the copy writes only the pixels, so the row padding compares equal.
	std::memcpy(download->Mapped().data(), guest->data(), guest->size() * sizeof(uint32_t));
	download->Flush(0, range.size);
	DownloadImage(*image, *download, 0, range.size, std::move(transfer));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = 0;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
	                                                 nullptr, 1, &barrier, 0, nullptr);
	const bool grid  = image->info.pixel_format == vk::Format::eR32G32Uint;
	const auto width = image->info.extent.width, height = image->info.extent.height;
	const auto layers = image->info.resources.layers;
	const auto pitch  = std::max(image->info.pitch, width);
	m_scheduler.DeferPriorityOperation([download, guest, range, op, flags, grid, width, height,
	                                    layers, pitch] {
		download->Invalidate(0, range.size);
		const auto* gpu    = reinterpret_cast<const uint32_t*>(download->Mapped().data());
		const auto  words  = guest->size();
		size_t      differ = 0, first = words;
		for (size_t i = 0; i < words; i++) {
			if (gpu[i] != (*guest)[i]) {
				first = differ++ == 0 ? i : first;
			}
		}
		// For a grid (two words a cell), where the first difference is.
		const uint64_t layer_words = words / std::max(layers, 1u);
		const auto     place =
            grid && differ != 0 && layer_words != 0
		            ? fmt::format(" (layer {}, row {}, cell {})", first / layer_words,
		                          (first % layer_words) / (static_cast<uint64_t>(pitch) * 2u),
		                          ((first % layer_words) % (static_cast<uint64_t>(pitch) * 2u)) / 2u)
		            : std::string();
		LOGF("FG %u grid 0x%012" PRIx64 "+0x%" PRIx64 " %ux%u layers %u pitch %u [%s]: the GPU's "
		     "pixels %s the guest bytes (%zu of %zu words differ%s)\n",
		     op, range.address, range.size, width, height, layers, pitch, flags.c_str(),
		     differ == 0 ? "match" : "differ from", differ, words,
		     differ == 0 ? ""
		                 : fmt::format(", the first at +0x{:x}{}: guest {:08x}, GPU {:08x}",
		                               first * 4, place, (*guest)[first], gpu[first])
		                       .c_str());
		if (!grid) {
			return;
		}
		LOGF("FG %u grid 0x%012" PRIx64 " lit cells/most lights per layer, guest:%s\n", op,
		     range.address,
		     LitCellsPerLayer(guest->data(), words, width, height, layers, pitch).c_str());
		if (differ != 0) {
			LOGF("FG %u grid 0x%012" PRIx64 " lit cells/most lights per layer, GPU:%s\n", op,
			     range.address, LitCellsPerLayer(gpu, words, width, height, layers, pitch).c_str());
		}
	});
}

// Diagnostics helpers (the frame graph's verification, the KYTY_TRACE_* traces): what the layers
// of an image hold, a glyph a layer.
namespace {

[[nodiscard]] float UnsignedSmallFloat(uint32_t bits, uint32_t mantissa_bits) {
	const uint32_t exponent = bits >> mantissa_bits;
	const uint32_t mantissa = bits & ((1u << mantissa_bits) - 1u);
	if (exponent == 31u) {
		return -1.0f; // infinity or NaN
	}
	const auto scale = static_cast<float>(1u << mantissa_bits);
	if (exponent == 0u) {
		return std::ldexp(static_cast<float>(mantissa) / scale, -14);
	}
	return std::ldexp(1.0f + static_cast<float>(mantissa) / scale, static_cast<int>(exponent) - 15);
}

[[nodiscard]] float HalfToFloat(uint32_t half) {
	const uint32_t exponent = (half >> 10u) & 0x1fu;
	const uint32_t mantissa = half & 0x3ffu;
	const float    sign     = (half & 0x8000u) != 0 ? -1.0f : 1.0f;
	if (exponent == 31u) {
		return std::numeric_limits<float>::quiet_NaN(); // infinity counts as not finite too
	}
	if (exponent == 0u) {
		return sign * std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
	}
	return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
	                         static_cast<int>(exponent) - 15);
}

// The words a texel of the float formats light data comes in takes; 0 for other formats.
[[nodiscard]] uint32_t FloatTexelWords(vk::Format format) {
	switch (format) {
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR32Sfloat: return 1;
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32G32Sfloat: return 2;
		case vk::Format::eR32G32B32A32Sfloat: return 4;
		default: return 0;
	}
}

[[nodiscard]] float Luminance(float r, float g, float b) {
	return 0.2126f * std::fabs(r) + 0.7152f * std::fabs(g) + 0.0722f * std::fabs(b);
}

// The light of a texel of such a format: its luminance (the first channel's magnitude for one
// and two channels), or a negative value when a channel, alpha too, is not finite.
[[nodiscard]] float TexelLight(vk::Format format, const uint32_t* word) {
	switch (format) {
		case vk::Format::eB10G11R11UfloatPack32: {
			const float r = UnsignedSmallFloat(word[0] & 0x7ffu, 6);
			const float g = UnsignedSmallFloat((word[0] >> 11u) & 0x7ffu, 6);
			const float b = UnsignedSmallFloat(word[0] >> 22u, 5);
			return r < 0.0f || g < 0.0f || b < 0.0f ? -1.0f : Luminance(r, g, b);
		}
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat: {
			const uint32_t       channels = format == vk::Format::eR16G16Sfloat ? 2u : 4u;
			std::array<float, 4> value {};
			for (uint32_t c = 0; c < channels; c++) {
				value[c] = HalfToFloat((word[c / 2u] >> (16u * (c % 2u))) & 0xffffu);
				if (!std::isfinite(value[c])) {
					return -1.0f;
				}
			}
			return channels == 2u ? std::fabs(value[0]) : Luminance(value[0], value[1], value[2]);
		}
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat: {
			const uint32_t channels = FloatTexelWords(format);
			for (uint32_t c = 0; c < channels; c++) {
				if (!std::isfinite(std::bit_cast<float>(word[c]))) {
					return -1.0f;
				}
			}
			const float r = std::bit_cast<float>(word[0]);
			return channels == 4u
			           ? Luminance(r, std::bit_cast<float>(word[1]), std::bit_cast<float>(word[2]))
			           : std::fabs(r);
		}
		default: return -1.0f;
	}
}

// The words a texel of a scanned image takes (TextureCache::ScanImageLocked): the float formats
// light data comes in, and other uncompressed formats of whole words, whose texels are only told
// zero or not; 0 for the rest (block-compressed, under a word).
[[nodiscard]] uint32_t ScanTexelWords(vk::Format format) {
	if (const auto words = FloatTexelWords(format); words != 0) {
		return words;
	}
	const auto extent = vk::blockExtent(format);
	const auto bytes  = vk::blockSize(format);
	if (extent[0] != 1 || extent[1] != 1 || bytes < 4u || bytes % 4u != 0) {
		return 0;
	}
	return bytes / 4u;
}

// Words in a guest layout, put into the host's (as ImageInfo::GetColorTransform's transfer
// transform uploads them): the guest's 10_11_11 keeps its 10-bit red in the low bits and its
// 11-bit blue in the high ones, the host's B10G11R11 the other way round
// (Guest10_11_11ToHost), so the guest's bytes decoded as the host format read wrong.
void GuestToHostLayout(uint32_t* words, size_t count, ColorTransform transform) {
	switch (transform) {
		case ColorTransform::Reverse10_11_11:
			for (size_t i = 0; i < count; i++) {
				words[i] = Guest10_11_11ToHost(words[i]);
			}
			break;
		case ColorTransform::SwapBgra16:
			for (size_t i = 0; i + 1u < count; i += 2u) {
				const uint32_t x = words[i];
				const uint32_t y = words[i + 1u];
				words[i]         = (y & 0xffffu) | (x & 0xffff0000u);
				words[i + 1u]    = (x & 0xffffu) | (y & 0xffff0000u);
			}
			break;
		case ColorTransform::None: break;
	}
}

// The bytes of mip 0 of every layer (or depth slice) of an image, tightly packed, as
// TextureCache::ScanImageLocked copies them; 0 when it does not scan the image.
[[nodiscard]] uint64_t ScanTexelBytes(const Image& image) {
	if (image.depth_id || image.backing.image == nullptr || image.backing.samples != 1 ||
	    image.info.IsDepth() || image.info.data.Empty()) {
		return 0;
	}
	const uint32_t texel_words = ScanTexelWords(image.backing.format);
	if (texel_words == 0) {
		return 0;
	}
	const uint64_t width       = std::max(image.backing.extent.width, 1u);
	const uint64_t height      = std::max(image.backing.extent.height, 1u);
	const uint64_t layers      = image.backing.image_type == vk::ImageType::e3D
	                                 ? std::max(image.backing.extent.depth, 1u)
	                                 : std::max(image.backing.layers, 1u);
	const uint64_t layer_bytes = width * height * texel_words * sizeof(uint32_t);
	if (layer_bytes == 0 || layer_bytes > UINT64_MAX / layers) {
		return 0;
	}
	return layer_bytes * layers;
}

[[nodiscard]] bool ZeroTexel(const uint32_t* word, uint32_t words) {
	for (uint32_t k = 0; k < words; k++) {
		if (word[k] != 0) {
			return false;
		}
	}
	return true;
}

// What a layer's texels hold.
struct LayerScan {
	size_t texels    = 0;
	size_t zero      = 0;
	size_t nonfinite = 0;
	double light     = 0.0; // of the finite non-zero texels, summed
};

[[nodiscard]] LayerScan ScanTexels(const uint32_t* words, size_t count, vk::Format format) {
	const uint32_t texel_words = FloatTexelWords(format);
	const uint32_t step        = std::max(texel_words, 1u);
	LayerScan      scan;
	for (size_t k = 0; k + step <= count; k += step) {
		scan.texels++;
		if (ZeroTexel(words + k, step)) {
			scan.zero++;
			continue;
		}
		if (texel_words == 0) {
			continue;
		}
		const float light = TexelLight(format, words + k);
		if (light < 0.0f) {
			scan.nonfinite++;
		} else {
			scan.light += light;
		}
	}
	return scan;
}

// One glyph a layer: '_' when every texel is zero. For the float formats light comes in: a digit,
// the decade of the mean light of its non-zero texels (5: 1 to 10, 4: 0.1 to 1, ... 0: below
// 1e-4, 9: 1e4 and up), the letter a-j instead when under a quarter of its texels are non-zero;
// '!' when some of its texels are not finite (infinity or NaN), '#' when every non-zero one is.
// '+' for a layer of another format that is not all zero.
[[nodiscard]] char LayerGlyph(const LayerScan& scan, bool float_format) {
	const size_t nonzero = scan.texels - scan.zero;
	if (nonzero == 0) {
		return '_';
	}
	if (!float_format) {
		return '+';
	}
	if (scan.nonfinite == nonzero) {
		return '#';
	}
	if (scan.nonfinite != 0) {
		return '!';
	}
	const double mean   = scan.light / static_cast<double>(nonzero);
	const int    decade = !std::isfinite(mean) ? 9
	                      : mean <= 0.0        ? 0
	                                           : static_cast<int>(std::floor(std::log10(mean))) + 5;
	return static_cast<char>((nonzero * 4u < scan.texels ? 'a' : '0') + std::clamp(decade, 0, 9));
}

constexpr const char* LayerGlyphLegend =
    "'_' all zero; else a digit for the decade of the mean light, 5 for 1 to 10, 4 for 0.1 to 1 "
    "...; a-j when mostly zero; '!' some texels not finite (infinity or NaN), '#' all of them; "
    "'+' not a float format";

// The words most of the texels that are not finite hold (their first word), up to `most`.
[[nodiscard]] std::string CommonNonFiniteWords(const uint32_t* words, size_t count,
                                               vk::Format format, size_t most) {
	const uint32_t                         step = std::max(FloatTexelWords(format), 1u);
	std::unordered_map<uint32_t, uint32_t> counts;
	for (size_t k = 0; k + step <= count; k += step) {
		if (!ZeroTexel(words + k, step) && TexelLight(format, words + k) < 0.0f) {
			counts[words[k]]++;
		}
	}
	std::vector<std::pair<uint32_t, uint32_t>> sorted(counts.begin(), counts.end());
	std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
		return a.second != b.second ? a.second > b.second : a.first < b.first;
	});
	std::string text;
	for (size_t i = 0; i < std::min(most, sorted.size()); i++) {
		text += fmt::format(" {:08x} x{}", sorted[i].first, sorted[i].second);
	}
	if (sorted.size() > most) {
		text += fmt::format(" (and {} other words)", sorted.size() - most);
	}
	return text;
}

// A glyph a layer (LayerGlyph), and for each run of layers that hold texels that are not finite,
// up to max_details, a line on the run's first layer: how many of its texels, and their words.
struct LayerSummary {
	std::string              glyphs;
	std::vector<std::string> details;
	uint32_t                 nonfinite_layers = 0;
	uint32_t                 runs             = 0;
};

[[nodiscard]] LayerSummary SummarizeLayers(const uint32_t* words, size_t count, uint32_t layers,
                                           vk::Format format, size_t max_details) {
	LayerSummary summary;
	if (layers == 0 || count < layers) {
		return summary;
	}
	const size_t layer_words  = count / layers;
	const bool   float_format = FloatTexelWords(format) != 0;
	summary.glyphs.reserve(layers);
	for (uint32_t layer = 0; layer < layers; layer++) {
		summary.glyphs += LayerGlyph(
		    ScanTexels(words + static_cast<size_t>(layer) * layer_words, layer_words, format),
		    float_format);
	}
	for (uint32_t first = 0; first < layers;) {
		const char glyph = summary.glyphs[first];
		if (glyph != '#' && glyph != '!') {
			first++;
			continue;
		}
		uint32_t last = first;
		while (last + 1u < layers && summary.glyphs[last + 1u] == glyph) {
			last++;
		}
		summary.runs++;
		summary.nonfinite_layers += last - first + 1u;
		if (summary.details.size() < max_details) {
			const auto* layer_first = words + static_cast<size_t>(first) * layer_words;
			const auto  scan        = ScanTexels(layer_first, layer_words, format);
			summary.details.push_back(fmt::format(
			    "layers {}-{} '{}': layer {} has {} of {} texels not finite, {} zero; the most "
			    "common words:{}",
			    first, last, glyph, first, scan.nonfinite, scan.texels, scan.zero,
			    CommonNonFiniteWords(layer_first, layer_words, format, 4)));
		}
		first = last + 1u;
	}
	return summary;
}

void LogGlyphLines(const std::string& prefix, const std::string& glyphs) {
	constexpr size_t Chunk = 192;
	for (size_t first = 0; first < glyphs.size(); first += Chunk) {
		const auto part = glyphs.substr(first, Chunk);
		LOGF("%s [%zu-%zu]: %s\n", prefix.c_str(), first, first + part.size() - 1u, part.c_str());
	}
}

void LogLayerDetails(const std::string& prefix, const LayerSummary& summary) {
	if (summary.nonfinite_layers == 0) {
		return;
	}
	LOGF("%s: %u of %zu layers hold texels that are not finite (%u runs)\n", prefix.c_str(),
	     summary.nonfinite_layers, summary.glyphs.size(), summary.runs);
	for (const auto& detail: summary.details) {
		LOGF("%s %s\n", prefix.c_str(), detail.c_str());
	}
}

void LogLayerSummary(const std::string& prefix, const LayerSummary& summary) {
	LogGlyphLines(prefix, summary.glyphs);
	LogLayerDetails(prefix, summary);
}

// The runs of layers whose glyph changed: " <first>-<last>:<before>><after>" each (up to 12
// glyphs a side), up to `most` runs; `changed` counts the layers.
[[nodiscard]] std::string ChangedLayers(const std::string& before, const std::string& after,
                                        size_t most, size_t& changed) {
	std::string text;
	size_t      runs = 0;
	changed          = 0;
	for (size_t i = 0; i < after.size();) {
		if (i < before.size() && before[i] == after[i]) {
			i++;
			continue;
		}
		size_t j = i;
		while (j < after.size() && (j >= before.size() || before[j] != after[j])) {
			j++;
		}
		changed += j - i;
		if (runs++ < most) {
			const auto shown = std::min<size_t>(j - i, 12);
			text += fmt::format(" {}-{}:{}>{}", i, j - 1,
			                    i < before.size() ? before.substr(i, shown) : std::string(),
			                    after.substr(i, shown));
		}
		i = j;
	}
	if (runs > most) {
		text += fmt::format(" (and {} more runs)", runs - most);
	}
	return text;
}

// KYTY_TRACE_VERIFY_MB (default 2048): how much the traces' guest-memory comparisons
// (TextureCache::TraceVerify) download in all, each range once. `peek` only says whether a claim
// would be granted.
bool TraceVerifyClaim(uint64_t address, uint64_t size, bool peek) {
	static std::mutex                              mutex;
	static std::set<std::pair<uint64_t, uint64_t>> claimed;
	static uint64_t                                used   = 0;
	static const uint64_t                          budget = [] {
        const char* value = std::getenv("KYTY_TRACE_VERIFY_MB");
        return (value != nullptr ? std::strtoull(value, nullptr, 10) : 2048ull) << 20u;
	}();
	std::scoped_lock lock {mutex};
	if (size == 0 || size > FrameGraph::MaxVerifyBytes || used + size > budget ||
	    claimed.contains({address, size})) {
		return false;
	}
	if (!peek) {
		claimed.emplace(address, size);
		used += size;
	}
	return true;
}

bool ClaimTraceVerify(uint64_t address, uint64_t size) {
	return TraceVerifyClaim(address, size, false);
}

} // namespace

void TextureCache::VerifyWithGuest(ImageId id, uint32_t op, const std::string& what) {
	CompareImageWithGuest(id, fmt::format("FG {}", op), what, true, &FrameGraph::ClaimVerifyTexture,
	                      &FrameGraph::ClaimVerifyLayers);
}

void TextureCache::TraceVerify(ImageId id, const std::string& label, const std::string& what) {
	CompareImageWithGuest(id, label, what, false, &ClaimTraceVerify, &ClaimTraceVerify);
}

void TextureCache::CompareImageWithGuest(ImageId id, const std::string& label,
                                         const std::string& what, bool frame_graph,
                                         bool (*claim)(uint64_t, uint64_t),
                                         bool (*claim_layers)(uint64_t, uint64_t)) {
	if (!id) {
		return;
	}
	// Before the lock: DiagnosticImageState takes it.
	const auto       state = DiagnosticImageState(id);
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(id);
	if (image == nullptr || image->depth_id || image->info.data.Empty() || image->info.IsDepth() ||
	    image->info.samples != 1 ||
	    image->info.metadata.compression != VideoOutCompression::Uncompressed) {
		return;
	}
	const auto range = image->info.data;
	const bool gpu   = image->IsGpuModified() || image->IsBufferModified() ||
	                 m_buffer_cache.IsRegionGpuModified(range.address, range.size);
	const auto layers  = image->info.resources.layers;
	const bool layered = layers >= 6u && !image->info.IsVolume() && range.size % layers == 0;
	// A linear image with a colour transform downloads through a scratch copy that does not keep
	// the padding between its rows: its comparison would always differ.
	const bool transformed_linear =
	    !image->info.IsTiled() && image->info.GetColorTransform() != ColorTransform::None;
	if ((gpu && !layered) || (!gpu && transformed_linear) || range.size % sizeof(uint32_t) != 0) {
		return;
	}
	// The download, and the copy of mip 0 an array's layers are told from.
	const uint64_t scan_bytes  = layered ? ScanTexelBytes(*image) : 0;
	const uint64_t claim_bytes = range.size + scan_bytes;
	const auto     claim_now   = gpu ? claim_layers : claim;
	// A frame capture claims each range once: no more work for the next bindings of it. A trace's
	// claims hold for the whole run: claimed only once the comparison is recorded.
	if (frame_graph ? !claim_now(range.address, claim_bytes)
	                : !TraceVerifyClaim(range.address, claim_bytes, true)) {
		return;
	}
	const bool tracked_dirty = image->IsCpuDirty();
	const bool cached_buffer = m_buffer_cache.IsRegionRegistered(range.address, range.size);
	if (!gpu) {
		// As the binding will see it (RebindImages refreshes it after this).
		RefreshImage(id);
		image = m_slot_images.try_get(id);
		if (image == nullptr || image->IsDefinitelyCpuDirty()) {
			return;
		}
	}
	auto transfer = BuildDownload(*image);
	if (!transfer.valid || transfer.depth_target ||
	    (!frame_graph && !claim_now(range.address, claim_bytes))) {
		return;
	}
	const auto& info        = image->info;
	const auto  description = fmt::format(
        "{} 0x{:012x}+0x{:x} {}x{}x{} layers {} mips {} fmt {} guest {} tile {} [{}{}{}] overlaps "
	     "{} ({} GPU-written){}{}",
        what, range.address, range.size, info.extent.width, info.extent.height, info.extent.depth,
        layers, info.resources.levels, static_cast<int>(info.pixel_format),
        static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
        gpu ? 'G' : '-', image->IsBufferModified() ? 'B' : '-', tracked_dirty ? 'C' : '-',
        state.overlaps, state.gpu_overlaps, cached_buffer ? ", a cached buffer covers it" : "",
        !gpu && image->IsMaybeCpuDirty() ? ", maybe CPU-written since (a write next to it)" : "");
	// An array's mip 0 as the GPU holds it, a glyph a layer (the float formats light data comes
	// in): its texels alone, without the guest layout's padding.
	if (layered) {
		(void)ScanImageLocked(*image, fmt::format("{} layers 0x{:012x}", label, range.address),
		                      FrameGraph::MaxVerifyBytes, nullptr,
		                      fmt::format("{} layers {}, mip 0 as the GPU holds it ({}):", label,
		                                  description, LayerGlyphLegend));
	}
	const size_t words = range.size / sizeof(uint32_t);
	// The guest bytes: what the GPU's copy of an image it did not write is compared with; for an
	// array it wrote, what guest memory holds beside what the GPU does.
	auto guest = std::make_shared<std::vector<uint32_t>>(words);
	if (!LibKernel::Memory::TryReadBacking(range.address, guest->data(), range.size)) {
		return;
	}
	auto download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         AllFlags, range.size);
	if (download->Mapped().data() == nullptr) {
		return;
	}
	// The guest bytes first: the copy writes only the texels, so the padding between them compares
	// equal.
	std::memcpy(download->Mapped().data(), guest->data(), range.size);
	download->Flush(0, range.size);
	DownloadImage(*image, *download, 0, range.size, std::move(transfer));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = 0;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
	                                                 nullptr, 1, &barrier, 0, nullptr);
	const auto format    = info.pixel_format;
	const auto transform = info.GetColorTransform();
	m_scheduler.DeferPriorityOperation([download, guest, range, words, label, description, layered,
	                                    layers, format, transform, gpu, frame_graph] {
		download->Invalidate(0, range.size);
		const auto* held   = reinterpret_cast<const uint32_t*>(download->Mapped().data());
		size_t      differ = 0, first = words, guest_zero = 0, gpu_zero = 0;
		// A 10_11_11 image's host blue has one mantissa bit fewer than the guest's: an upload
		// read back differs there (Guest10_11_11ToHost), which is not a difference.
		const uint32_t compared = transform == ColorTransform::Reverse10_11_11 ? ~(1u << 21u) : ~0u;
		const auto     same = [&](size_t i) { return ((held[i] ^ (*guest)[i]) & compared) == 0; };
		for (size_t i = 0; i < words; i++) {
			guest_zero += (*guest)[i] == 0 ? 1u : 0u;
			gpu_zero += held[i] == 0 ? 1u : 0u;
			if (!same(i)) {
				first = differ++ == 0 ? i : first;
			}
		}
		const bool stale = !gpu && differ != 0;
		if (stale) {
			LOGF("%s stale %s: %zu of %zu words differ, the first at +0x%zx (guest %08x, GPU "
			     "%08x); zero words: guest %zu, GPU %zu\n",
			     label.c_str(), description.c_str(), differ, words, first * 4u, (*guest)[first],
			     held[first], guest_zero, gpu_zero);
		}
		if (layered && (gpu || stale)) {
			// Which layers the GPU holds as guest memory does: for one the GPU wrote, the layers it
			// never wrote since their upload (or wrote the same bytes into).
			const size_t layer_words = words / layers;
			std::string  glyphs;
			glyphs.reserve(layers);
			for (uint32_t layer = 0; layer < layers; layer++) {
				const auto at    = static_cast<size_t>(layer) * layer_words;
				bool       equal = true;
				for (size_t i = at; i < at + layer_words && equal; i++) {
					equal = same(i);
				}
				glyphs += equal ? '=' : 'x';
			}
			LOGF("%s %s %s: '=' the GPU holds the layer as guest memory does, 'x' it does not:\n",
			     label.c_str(), gpu ? "guest-equal layers" : "stale layers", description.c_str());
			LogGlyphLines(fmt::format("{} {} 0x{:012x}", label,
			                          gpu ? "guest-equal layers" : "stale layers", range.address),
			              glyphs);
		}
		if (layered && gpu) {
			// What guest memory holds, padding included (its texels in the host's bit layout).
			GuestToHostLayout(guest->data(), words, transform);
			const auto summary = SummarizeLayers(guest->data(), words, layers, format, 6);
			LOGF("%s guest layers 0x%012" PRIx64
			     ", as guest memory holds them (padding included):\n",
			     label.c_str(), range.address);
			LogLayerSummary(fmt::format("{} guest layers 0x{:012x}", label, range.address),
			                summary);
		}
		if (frame_graph) {
			FrameGraph::CountVerify(true, range.size, stale);
		}
	});
}

bool TextureCache::ScanImageLocked(Image& image, const std::string& prefix, uint64_t max_bytes,
                                   std::shared_ptr<TraceHistory> history,
                                   const std::string&            header) {
	if (image.depth_id || image.backing.image == nullptr || image.backing.samples != 1 ||
	    image.info.IsDepth() || image.info.data.Empty()) {
		return false;
	}
	const auto     format = image.backing.format;
	const uint64_t bytes  = ScanTexelBytes(image);
	if (bytes == 0 || bytes > max_bytes) {
		return false;
	}
	// The host image's own extent: a copy region must lie within it.
	const uint32_t width  = std::max(image.backing.extent.width, 1u);
	const uint32_t height = std::max(image.backing.extent.height, 1u);
	const bool     volume = image.backing.image_type == vk::ImageType::e3D;
	const uint32_t layers =
	    volume ? std::max(image.backing.extent.depth, 1u) : std::max(image.backing.layers, 1u);
	auto download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                         AllFlags, bytes);
	if (download->Mapped().data() == nullptr) {
		return false;
	}
	vk::BufferImageCopy region {};
	region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, volume ? 1u : layers};
	region.imageExtent      = vk::Extent3D {width, height, volume ? layers : 1u};
	image.Download(std::span {&region, 1u}, download->Handle(), 0, bytes);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = 0;
	barrier.size                = bytes;
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
	                                                 nullptr, 1, &barrier, 0, nullptr);
	m_scheduler.DeferPriorityOperation([download, bytes, layers, format, prefix, history, header] {
		download->Invalidate(0, bytes);
		const auto* words   = reinterpret_cast<const uint32_t*>(download->Mapped().data());
		const auto  summary = SummarizeLayers(words, bytes / sizeof(uint32_t), layers, format, 6);
		if (!header.empty()) {
			LOGF("%s\n", header.c_str());
		}
		if (history == nullptr) {
			LogLayerSummary(prefix, summary);
			return;
		}
		std::string before;
		{
			std::scoped_lock lock {history->mutex};
			before = std::exchange(history->glyphs, summary.glyphs);
		}
		if (before.empty()) {
			LogLayerSummary(prefix, summary);
			return;
		}
		size_t     changed = 0;
		const auto runs    = ChangedLayers(before, summary.glyphs, 40, changed);
		LOGF("%s: %zu of %zu layers changed%s%s; %u hold texels that are not finite\n",
		     prefix.c_str(), changed, summary.glyphs.size(), changed != 0 ? ":" : "", runs.c_str(),
		     summary.nonfinite_layers);
		if (changed > 96) {
			LogGlyphLines(prefix, summary.glyphs);
		}
		if (changed != 0) {
			LogLayerDetails(prefix, summary);
		}
	});
	return true;
}

bool TextureCache::DumpImage(ImageId id, uint32_t capture, const std::string& name,
                             const std::string& description) {
	if (!id) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(id);
	return image != nullptr && DumpImageLocked(*image, false, capture, name, description);
}

static bool SparseHoles() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_HOLES");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("Sparse depth: the unmapped blocks of partly mapped depth images %s "
		     "(KYTY_SPARSE_HOLES)\n",
		     on ? "pass the depth comparison that samples them" : "keep what was drawn into them");
		return on;
	}();
	return enabled;
}

void TextureCache::ResetUnmappedDepth(ImageId id, float empty) {
	if (!id || !SparseHoles()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	auto*            found = m_slot_images.try_get(id);
	if (found == nullptr || found->sparse_checked_serial == found->content_serial) {
		return;
	}
	auto&       image             = *found;
	const auto& info              = image.info;
	image.sparse_checked_serial   = image.content_serial;
	const auto         format     = image.backing.format;
	const uint32_t     bytes      = format == vk::Format::eD16Unorm    ? 2u
	                                : format == vk::Format::eD32Sfloat ? 4u
	                                                                   : 0u;
	constexpr uint64_t BlockBytes = 0x10000u;
	if (bytes == 0 || image.backing.image == nullptr || image.depth_id || info.data.Empty() ||
	    info.tile_mode != Prospero::TileMode::kDepth || info.HasStencil() || info.IsVolume() ||
	    info.resources.levels != 1 || info.resources.layers != 1 || image.backing.samples != 1 ||
	    image.backing.mip_levels != 1 || (info.data.address % BlockBytes) != 0 ||
	    !(image.backing.usage & vk::ImageUsageFlagBits::eTransferDst)) {
		return;
	}
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	if (!Libs::LibKernel::Memory::CommittedSpans(info.data.address, info.data.size, 1u << 20u,
	                                             &spans)) {
		return;
	}
	uint64_t committed = 0;
	for (const auto& span: spans) {
		committed += span.second;
	}
	if (committed >= info.data.size) {
		return; // all mapped: the common case
	}
	TileBlockLayout block {};
	if (!TileGetBlockLayout(TileBlockFamily::Depth64KB, bytes, block) ||
	    block.block_size != BlockBytes || block.block_depth != 1) {
		return;
	}
	const uint32_t width          = std::max(image.backing.extent.width, 1u);
	const uint32_t height         = std::max(image.backing.extent.height, 1u);
	const uint32_t blocks_per_row = (width + block.block_width - 1u) / block.block_width;
	const uint32_t block_rows     = (height + block.block_height - 1u) / block.block_height;
	const uint64_t blocks         = uint64_t {blocks_per_row} * block_rows;
	if (blocks * BlockBytes > info.data.size) {
		return; // not the layout assumed here (64 KB blocks, row by row)
	}
	// Whether each block is mapped: any byte of it is (a block partly mapped keeps its texels).
	std::vector<bool> mapped(blocks, false);
	for (const auto& [start, size]: spans) {
		const uint64_t first = (start - info.data.address) / BlockBytes;
		const uint64_t end   = std::min<uint64_t>(
            (start + size - info.data.address + BlockBytes - 1u) / BlockBytes, blocks);
		for (uint64_t b = first; b < end; b++) {
			mapped[b] = true;
		}
	}
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         largest = 0;
	uint64_t                         holes   = 0;
	for (uint32_t row = 0; row < block_rows; row++) {
		for (uint32_t column = 0; column < blocks_per_row;) {
			if (mapped[uint64_t {row} * blocks_per_row + column]) {
				column++;
				continue;
			}
			uint32_t end = column;
			while (end < blocks_per_row && !mapped[uint64_t {row} * blocks_per_row + end]) {
				end++;
			}
			holes += end - column;
			const uint32_t      x = column * block.block_width;
			const uint32_t      y = row * block.block_height;
			vk::BufferImageCopy region {};
			region.imageSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
			region.imageOffset = vk::Offset3D {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
			region.imageExtent = vk::Extent3D {std::min(end * block.block_width, width) - x,
			                                   std::min(block.block_height, height - y), 1};
			largest            = std::max<uint64_t>(largest, uint64_t {region.imageExtent.width} *
			                                                     region.imageExtent.height * bytes);
			regions.push_back(region);
			column = end;
		}
	}
	if (regions.empty()) {
		return;
	}
	// One far-depth source for every region (each reads it from its start).
	const uint64_t source_size = Common::AlignUp(largest, uint64_t {4});
	auto           source      = std::make_shared<Buffer>(
        m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, source_size);
	m_scheduler.EndRendering();
	const uint32_t empty_word =
	    bytes == 2u ? (empty >= 1.0f ? 0xffffffffu : 0u) : std::bit_cast<uint32_t>(empty);
	m_scheduler.Current().Recorder().fillBuffer(source->Handle(), 0, source_size, empty_word);
	image.Upload(regions, source->Handle(), 0, source_size);
	m_scheduler.DeferRelease([source] {});
	image.content_serial++;
	image.sparse_checked_serial = image.content_serial;
	static std::mutex                   logged_mutex;
	static std::unordered_set<uint64_t> logged;
	std::lock_guard                     logged_lock(logged_mutex);
	if (logged.size() < 32 &&
	    logged.insert(info.data.address ^ (uint64_t {width} << 40u) ^ (uint64_t {holes} << 20u))
	        .second) {
		LOGF("Sparse depth: 0x%012" PRIx64 " %ux%u: %" PRIu64 " of %" PRIu64
		     " 64 KB blocks unmapped, reset to depth %.1f (%zu regions)\n",
		     info.data.address, width, height, holes, blocks, static_cast<double>(empty),
		     regions.size());
	}
}

bool TextureCache::DumpImageLocked(Image& target, bool trace, uint32_t capture,
                                   const std::string& name, const std::string& description) {
	auto* image = &target;
	// An image with a depth twin keeps its contents there (UploadImage), as ScanImageLocked
	// leaves out.
	if (image->backing.image == nullptr || image->backing.samples != 1 || image->depth_id) {
		return false;
	}
	const auto format      = image->backing.format;
	bool       depth       = false;
	const auto texel_bytes = ImageDump::TexelBytes(format, depth);
	uint32_t   block_bytes = 0;
	const auto dxgi_format = texel_bytes == 0 ? ImageDump::BlockFormat(format, block_bytes) : 0u;
	if (dxgi_format != 0) {
		return DumpBlocksLocked(*image, trace, capture, name, description, dxgi_format,
		                        block_bytes);
	}
	if (texel_bytes == 0) {
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("FG dump: %s 0x%012" PRIx64 " not dumped: format %d is not decoded\n",
			     description.c_str(), image->info.data.address, static_cast<int>(format));
		}
		return false;
	}
	const auto& backing = image->backing;
	const bool  volume  = backing.image_type == vk::ImageType::e3D;
	const auto  levels  = std::max(backing.mip_levels, 1u);
	const auto  width   = std::max(backing.extent.width, 1u);
	const auto  height  = std::max(backing.extent.height, 1u);
	const auto  layers  = volume ? 1u : std::max(backing.layers, 1u);
	// The mip closest above the size shown, every step-th row and column of it (with
	// KYTY_DUMP_IMAGES_MIPS, every mip, each the same way).
	const bool     tiled_sheet = layers > 1 || (volume && backing.extent.depth > 1);
	const uint32_t side =
	    tiled_sheet ? std::max(ImageDump::MaxSide() / 2u, 64u) : ImageDump::MaxSide();
	uint32_t first = 0;
	while (first + 1 < levels && std::max(width >> (first + 1), height >> (first + 1)) >= side) {
		first++;
	}
	const auto budget = ImageDump::MaxBytes();
	struct MipPlan {
		uint32_t width = 0, height = 0, tiles = 0, all_tiles = 0, step = 1, rows = 0;
		uint64_t row_pitch = 0, tile_pitch = 0;
	};
	const auto plan_mip = [&](uint32_t mip) {
		MipPlan plan;
		plan.width      = std::max(width >> mip, 1u);
		plan.height     = std::max(height >> mip, 1u);
		plan.all_tiles  = volume ? std::max(backing.extent.depth >> mip, 1u) : layers;
		plan.step       = (std::max(plan.width, plan.height) + side - 1u) / side;
		plan.rows       = (plan.height + plan.step - 1u) / plan.step;
		plan.row_pitch  = plan.step == 1 ? uint64_t {plan.width} * texel_bytes
		                                 : Common::AlignUp(uint64_t {plan.width} * texel_bytes, 16);
		plan.tile_pitch = Common::AlignUp(plan.row_pitch * plan.rows, 16);
		if (plan.tile_pitch == 0 || plan.tile_pitch > budget) {
			return plan; // no tiles: not dumped
		}
		// Within the readback budget, and at most 16M texels shown (the decode keeps 16 bytes
		// each).
		const uint64_t shown = uint64_t {(plan.width + plan.step - 1u) / plan.step} * plan.rows;
		plan.tiles           = static_cast<uint32_t>(
            std::min<uint64_t>({uint64_t {plan.all_tiles}, budget / plan.tile_pitch,
		                                  std::max<uint64_t>((16ull << 20u) / shown, 1u)}));
		return plan;
	};
	if (plan_mip(first).tiles == 0) {
		return false;
	}
	if (trace ? !ImageDump::ClaimTrace(image->info.data.address, image->content_serial)
	          : !ImageDump::Claim(capture, image->info.data.address, image->content_serial)) {
		return false;
	}
	const bool     all_mips = !trace && ImageDump::AllMips();
	const uint32_t from     = all_mips ? 0u : first;
	const uint32_t to       = all_mips ? levels : first + 1u;
	const auto  aspect = depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
	const auto& info   = image->info;
	m_scheduler.EndRendering();
	bool dumped = false;
	for (uint32_t mip = from; mip < to; mip++) {
		const auto plan = plan_mip(mip);
		if (plan.tiles == 0) {
			continue;
		}
		const uint64_t bytes = plan.tile_pitch * plan.tiles;
		auto download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         AllFlags, bytes);
		if (download->Mapped().data() == nullptr) {
			break;
		}
		std::vector<vk::BufferImageCopy> regions;
		regions.reserve(static_cast<size_t>(plan.tiles) * (plan.step == 1 ? 1u : plan.rows));
		for (uint32_t tile = 0; tile < plan.tiles; tile++) {
			vk::BufferImageCopy region {};
			region.imageSubresource = {aspect, mip, volume ? 0u : tile, 1};
			if (plan.step == 1) {
				region.bufferOffset = tile * plan.tile_pitch;
				region.imageOffset  = vk::Offset3D {0, 0, volume ? static_cast<int32_t>(tile) : 0};
				region.imageExtent  = vk::Extent3D {plan.width, plan.height, 1};
				regions.push_back(region);
				continue;
			}
			for (uint32_t row = 0; row < plan.rows; row++) {
				region.bufferOffset = tile * plan.tile_pitch + row * plan.row_pitch;
				region.imageOffset  = vk::Offset3D {0, static_cast<int32_t>(row * plan.step),
                                                   volume ? static_cast<int32_t>(tile) : 0};
				region.imageExtent  = vk::Extent3D {plan.width, 1, 1};
				regions.push_back(region);
			}
		}
		m_scheduler.Current().MarkTransfer(
		    CommandBufferDebugOp::InternalWork, image->info.data.address, 0,
		    static_cast<uint32_t>(InternalWork::ImageDump),
		    static_cast<uint32_t>(std::min<uint64_t>(bytes, UINT32_MAX)), plan.tiles);
		image->Download(regions, download->Handle(), 0, bytes);
		vk::BufferMemoryBarrier barrier {};
		barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = download->Handle();
		barrier.offset              = 0;
		barrier.size                = bytes;
		m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
		                                                 nullptr, 1, &barrier, 0, nullptr);
		ImageDump::Layout layout;
		layout.format      = format;
		layout.row_texels  = plan.width;
		layout.rows        = plan.rows;
		layout.tiles       = plan.tiles;
		layout.step        = plan.step;
		layout.row_pitch   = plan.row_pitch;
		layout.tile_pitch  = plan.tile_pitch;
		layout.texel_bytes = texel_bytes;
		layout.depth       = depth;
		layout.cubes       = !volume && layers >= 6 && layers % 6 == 0;
		auto file = fmt::format("{}_{:012x}_{}x{}", name, info.data.address, width, height);
		if (layers > 1 || volume) {
			file += fmt::format("x{}", volume ? backing.extent.depth : layers);
		}
		file += fmt::format("_m{}_f{}", mip, static_cast<int>(format));
		auto text = fmt::format(
		    "{} 0x{:012x}+0x{:x} {}x{}x{} layers {} mips {} fmt {} guest {} tile {} [{}{}{}], mip "
		    "{}{}",
		    description, info.data.address, info.data.size, info.extent.width, info.extent.height,
		    info.extent.depth, info.resources.layers, info.resources.levels,
		    static_cast<int>(info.pixel_format), static_cast<uint32_t>(info.guest_format),
		    static_cast<uint32_t>(info.tile_mode), image->IsGpuModified() ? 'G' : '-',
		    image->IsBufferModified() ? 'B' : '-', image->IsCpuDirty() ? 'C' : '-', mip,
		    plan.tiles < plan.all_tiles
		        ? fmt::format(", the first {} of {} (the size limits)", plan.tiles, plan.all_tiles)
		        : std::string());
		m_scheduler.DeferPriorityOperation(
		    [download, bytes, layout, file = std::move(file), text = std::move(text)]() mutable {
			    download->Invalidate(0, bytes);
			    const auto*          mapped = download->Mapped().data();
			    std::vector<uint8_t> copy(mapped, mapped + bytes);
			    ImageDump::Write(std::move(file), std::move(text), layout, std::move(copy));
		    });
		dumped = true;
	}
	return dumped;
}

// Round 109: a block-compressed image, each mip of its first layer (or slice) as the GPU holds
// its blocks, within the readback budget.
bool TextureCache::DumpBlocksLocked(Image& image, bool trace, uint32_t capture,
                                    const std::string& name, const std::string& description,
                                    uint32_t dxgi_format, uint32_t block_bytes) {
	if (trace) {
		return false;
	}
	const auto& backing = image.backing;
	const auto& info    = image.info;
	const auto  levels  = std::max(backing.mip_levels, 1u);
	const auto  width   = std::max(backing.extent.width, 1u);
	const auto  height  = std::max(backing.extent.height, 1u);
	const auto  budget  = ImageDump::MaxBytes();
	if (!ImageDump::Claim(capture, info.data.address, image.content_serial)) {
		return false;
	}
	// From the first mip no longer than KYTY_DUMP_IMAGES_SIZE on its longer side, to the last.
	uint32_t first = 0;
	while (first + 1 < levels && std::max(width >> first, height >> first) > ImageDump::MaxSide()) {
		first++;
	}
	m_scheduler.EndRendering();
	uint64_t total = 0;
	for (uint32_t mip = first; mip < levels; mip++) {
		const auto     mip_width  = std::max(width >> mip, 1u);
		const auto     mip_height = std::max(height >> mip, 1u);
		const uint64_t bytes =
		    uint64_t {(mip_width + 3u) / 4u} * ((mip_height + 3u) / 4u) * block_bytes;
		if (bytes == 0 || total + bytes > budget) {
			break;
		}
		auto download = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         AllFlags, bytes);
		if (download->Mapped().data() == nullptr) {
			break;
		}
		total += bytes;
		vk::BufferImageCopy region {};
		region.imageSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, 1};
		region.imageExtent      = vk::Extent3D {mip_width, mip_height, 1};
		m_scheduler.Current().MarkTransfer(
		    CommandBufferDebugOp::InternalWork, info.data.address, 0,
		    static_cast<uint32_t>(InternalWork::ImageDump),
		    static_cast<uint32_t>(std::min<uint64_t>(bytes, UINT32_MAX)), 1);
		const std::array<vk::BufferImageCopy, 1> regions {region};
		image.Download(regions, download->Handle(), 0, bytes);
		vk::BufferMemoryBarrier barrier {};
		barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = download->Handle();
		barrier.offset              = 0;
		barrier.size                = bytes;
		m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                                                 vk::PipelineStageFlagBits::eHost, {}, 0,
		                                                 nullptr, 1, &barrier, 0, nullptr);
		auto file = fmt::format("{}_{:012x}_{}x{}_m{}_f{}", name, info.data.address, width, height,
		                        mip, static_cast<int>(backing.format));
		auto text = fmt::format(
		    "{} 0x{:012x}+0x{:x} {}x{}x{} layers {} mips {} fmt {} guest {} tile {} [{}{}{}], mip "
		    "{} of layer 0",
		    description, info.data.address, info.data.size, info.extent.width, info.extent.height,
		    info.extent.depth, info.resources.layers, info.resources.levels,
		    static_cast<int>(info.pixel_format), static_cast<uint32_t>(info.guest_format),
		    static_cast<uint32_t>(info.tile_mode), image.IsGpuModified() ? 'G' : '-',
		    image.IsBufferModified() ? 'B' : '-', image.IsCpuDirty() ? 'C' : '-', mip);
		m_scheduler.DeferPriorityOperation([download, bytes, file = std::move(file),
		                                    text = std::move(text), dxgi_format, block_bytes,
		                                    mip_width, mip_height]() mutable {
			download->Invalidate(0, bytes);
			const auto*          mapped = download->Mapped().data();
			std::vector<uint8_t> copy(mapped, mapped + bytes);
			ImageDump::WriteBlocks(std::move(file), std::move(text), dxgi_format, block_bytes,
			                       mip_width, mip_height, std::move(copy));
		});
	}
	return total != 0;
}

bool TextureCache::TraceScanImage(ImageId id, const std::string& prefix, uint64_t max_bytes) {
	if (!id) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(id);
	return image != nullptr && ScanImageLocked(*image, prefix, max_bytes, nullptr, std::string());
}

bool TextureCache::ScanNonFiniteGpu(ImageId id, NonFiniteDone done) {
	if (!id) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(id);
	return image != nullptr && ScanNonFiniteGpuLocked(*image, std::move(done), GpuScanLimit());
}

uint64_t TextureCache::ContentSerial(ImageId id) {
	if (!id) {
		return 0;
	}
	std::scoped_lock lock {m_lock};
	const auto*      image = m_slot_images.try_get(id);
	return image != nullptr ? image->content_serial : 0;
}

bool TextureCache::ScanNonFiniteGpuLocked(Image& image, NonFiniteDone done, uint64_t max_bytes,
                                          bool scrub) {
	if (!m_scheduler.Active()) {
		return false;
	}
	auto           layout = NonFiniteScanLayout(image, GpuScan::MaxCounters);
	const uint64_t total  = layout.bytes;
	if (total == 0 || total > max_bytes) {
		return false;
	}
	scrub = scrub && layout.packing != NanScanner::Packing::Float32 &&
	        (image.backing.usage & vk::ImageUsageFlagBits::eTransferDst);
	const auto&     backing = image.backing;
	const uint32_t  layers  = std::max(backing.layers, 1u);
	const uint32_t  mips    = std::max(backing.mip_levels, 1u);
	NonFiniteCounts result;
	result.address      = image.info.data.address;
	result.width        = backing.extent.width;
	result.height       = backing.extent.height;
	result.layers       = layers;
	result.mips         = mips;
	result.format       = static_cast<int>(backing.format);
	result.guest_format = static_cast<uint32_t>(image.info.guest_format);
	result.words        = layout.words;
	const auto& regions = layout.regions;
	if (m_gpu_scan == nullptr) {
		m_gpu_scan = std::make_unique<GpuScan>(m_graphics, m_scheduler);
	}
	auto& scan = *m_gpu_scan;
	if (!scan.scanner.Valid()) {
		return false;
	}
	constexpr auto StorageFlags = vk::BufferUsageFlagBits::eStorageBuffer |
	                              vk::BufferUsageFlagBits::eTransferDst |
	                              vk::BufferUsageFlagBits::eShaderDeviceAddress;
	if (scan.scratch == nullptr || scan.scratch->Size() < total) {
		// A scrub copies the scratch back into the image (StorageFlags | eTransferSrc below).
		// A buffer still in use stays alive in the callbacks of the scans that use it.
		scan.scratch =
		    std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		                             StorageFlags | vk::BufferUsageFlagBits::eTransferSrc,
		                             Common::AlignUp(total, uint64_t {4} << 20u));
	}
	scan.last_frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (scan.counts == nullptr) {
		scan.counts = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		                                       StorageFlags | vk::BufferUsageFlagBits::eTransferSrc,
		                                       uint64_t {GpuScan::MaxCounters} * sizeof(uint32_t));
	}
	const uint64_t count_bytes = uint64_t {layers} * mips * sizeof(uint32_t);
	// A full ring waits for the GPU (and the callbacks) to be done with its oldest part.
	const auto [mapped, offset] = scan.ring->Map(count_bytes, 64);
	if (mapped == nullptr) {
		return false;
	}
	scan.ring->Commit();
	// The copy waits for the image's last writers, and for the last scan's reads of the scratch.
	m_scheduler.Current().MarkTransfer(CommandBufferDebugOp::InternalWork, image.info.data.address,
	                                   0, static_cast<uint32_t>(InternalWork::NanScan),
	                                   static_cast<uint32_t>(std::min<uint64_t>(total, UINT32_MAX)),
	                                   layers);
	image.Download(regions, scan.scratch->Handle(), 0, total);
	const auto        recorder = m_scheduler.Current().Recorder();
	vk::MemoryBarrier all {};
	all.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	all.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eAllCommands, {}, 1, &all, 0, nullptr, 0,
	                         nullptr);
	recorder.fillBuffer(scan.counts->Handle(), 0, count_bytes, 0);
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eComputeShader, {}, 1, &all, 0, nullptr, 0,
	                         nullptr);
	const auto scratch_address = scan.scratch->BufferDeviceAddress();
	const auto counts_address  = scan.counts->BufferDeviceAddress();
	for (uint32_t mip = 0; mip < mips; mip++) {
		scan.scanner.Scan(recorder, scratch_address + regions[mip].bufferOffset,
		                  counts_address + uint64_t {mip} * layers * sizeof(uint32_t),
		                  result.words[mip], layers, layout.packing, scrub);
	}
	if (scrub) {
		// The scrubbed words back into the image (Upload's barrier waits for the scan's writes).
		image.Upload(regions, scan.scratch->Handle(), 0, total);
		image.content_serial++;
	}
	vk::MemoryBarrier counted {};
	counted.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	counted.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                         vk::PipelineStageFlagBits::eTransfer, {}, 1, &counted, 0, nullptr, 0,
	                         nullptr);
	recorder.copyBuffer(scan.counts->Handle(), scan.ring->Handle(),
	                    vk::BufferCopy {0, offset, count_bytes});
	vk::BufferMemoryBarrier landed {};
	landed.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	landed.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	landed.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	landed.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	landed.buffer              = scan.ring->Handle();
	landed.offset              = offset;
	landed.size                = count_bytes;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                         {}, 0, nullptr, 1, &landed, 0, nullptr);
	m_scheduler.DeferPriorityOperation(
	    [ring = scan.ring, scratch = scan.scratch, counts = scan.counts, mapped, offset,
	     count_bytes, result = std::move(result), done = std::move(done)]() mutable {
		    ring->Invalidate(offset, count_bytes);
		    const auto* words = reinterpret_cast<const uint32_t*>(mapped);
		    result.counts.assign(words, words + count_bytes / sizeof(uint32_t));
		    done(std::move(result));
	    });
	return true;
}

// KYTY_NAN_SCRUB (default on; 0 turns it off): the NaN check also rewrites what it finds in the
// B10G11R11 and half-float arrays it checks (light probes): a NaN to zero, an infinity to the
// largest finite value. A probe that holds them shades the surfaces that reflect it black or with
// NaN, which the frame's temporal and blur passes then spread, and a probe relit from its last
// value keeps them for good. On the guest GPU a value out of these formats' range is stored as
// their largest one instead.
static bool NanScrub() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NAN_SCRUB");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("TextureCache: NaN scrub %s (KYTY_NAN_SCRUB): the NaN check %s\n", on ? "on" : "off",
		     on ? "replaces the texels that are not finite it finds (NaN with zero, infinity with "
		          "the largest value)"
		        : "only reports the texels that are not finite");
		return on;
	}();
	return enabled;
}

void TextureCache::NoteNanCheckCandidate(const Image& image) {
	if (image.backing.layers < 6u || image.backing.image_type != vk::ImageType::e2D ||
	    image.backing.samples != 1 || image.depth_id || !NonFinitePacking(image.backing.format)) {
		return;
	}
	auto& candidate = m_nan_check->candidates[image.self_id];
	if (candidate.address != image.info.data.address) {
		candidate         = {};
		candidate.address = image.info.data.address;
	}
}

void TextureCache::RunNanCheck() {
	// KYTY_BISECT's nancheck:off: no image is checked (or scrubbed) while the step is on.
	if (Bisect::NanCheckOff()) {
		return;
	}
	const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	// One image a frame end: a copy of its mips and a scan, on the GPU.
	bool scanned = false;
	for (auto it = m_nan_check->candidates.begin(); it != m_nan_check->candidates.end();) {
		auto* image = m_slot_images.try_get(it->first);
		if (image == nullptr || !image->registered ||
		    image->info.data.address != it->second.address) {
			it = m_nan_check->candidates.erase(it);
			continue;
		}
		auto&      candidate = it->second;
		const bool due = !candidate.checked || (frame >= candidate.frame + NanCheck::Interval &&
		                                        image->content_serial != candidate.content);
		// An image its next use uploads anew is checked after that upload.
		if (scanned || candidate.rejected || !due || image->IsCpuDirty() ||
		    image->IsBufferModified()) {
			++it;
			continue;
		}
		const auto bytes = NonFiniteScanLayout(*image, GpuScan::MaxCounters).bytes;
		if (bytes > m_nan_check->max_bytes) {
			LOGF("TextureCache: NaN check skips 0x%012" PRIx64 " %ux%ux%u mips %u fmt %d: %" PRIu64
			     " MB in all its mips, over KYTY_NAN_CHECK_MB=%" PRIu64 "\n",
			     image->info.data.address, image->backing.extent.width,
			     image->backing.extent.height, image->backing.layers, image->backing.mip_levels,
			     static_cast<int>(image->backing.format), (bytes + (1u << 20u) - 1u) >> 20u,
			     m_nan_check->max_bytes >> 20u);
		}
		if (bytes == 0 || bytes > m_nan_check->max_bytes ||
		    !ScanNonFiniteGpuLocked(
		        *image,
		        [results = m_nan_check->results, frame](NonFiniteCounts&& counts) {
			        ReportNanCheck(*results, frame, counts);
		        },
		        m_nan_check->max_bytes, NanScrub())) {
			candidate.rejected = true;
			++it;
			continue;
		}
		candidate.frame   = frame;
		candidate.content = image->content_serial;
		candidate.checked = true;
		scanned           = true;
		++it;
	}
}

void TextureCache::TraceUploadScan(ImageId id) {
	const auto serial = m_trace->images[id].serial;
	const auto frame  = m_graphics.presented_frames.load(std::memory_order_relaxed);
	(void)ScanNonFiniteGpuLocked(
	    m_slot_images[id],
	    [serial, frame](NonFiniteCounts&& counts) {
		    const auto found = FindNonFiniteLayers(counts);
		    LOGF("Trace image #%u frame %" PRIu64 " after its upload, on the GPU: %s%s%s\n", serial,
		         frame, DescribeNonFinite(counts, found).c_str(),
		         found.count != 0 ? ": layers" : "",
		         found.count != 0 ? SetRuns(found.layers, 24).c_str() : "");
	    },
	    GpuScanLimit());
}

bool TextureCache::TracesImage(ImageId id) const {
	return m_trace != nullptr && m_trace->images.contains(id);
}

void TextureCache::TraceImageCreated(ImageId id) {
	if (!m_trace->Matches(m_slot_images[id].info)) {
		return;
	}
	auto& followed  = m_trace->images[id];
	followed        = {};
	followed.serial = ++m_trace->serials;
	const auto& info = m_slot_images[id].info;
	TraceImageEvent(id,
	                fmt::format("created; its memory: {}", LibKernel::Memory::DescribeMappings(
	                                                           info.data.address, info.data.size)),
	                true);
}

void TextureCache::TraceImageEvent(ImageId id, const std::string& event, bool guest_glyphs) {
	if (m_trace == nullptr) {
		return;
	}
	const auto found = m_trace->images.find(id);
	if (found == m_trace->images.end()) {
		return;
	}
	auto&       followed = found->second;
	const auto& image    = m_slot_images[id];
	const auto& info     = image.info;
	LOGF("Trace image #%u 0x%012" PRIx64 "+0x%" PRIx64 " %ux%u layers %u mips %u fmt %d tile %u "
	     "frame %" PRIu64 " [%c%c%c]: %s\n",
	     followed.serial, info.data.address, info.data.size, info.extent.width, info.extent.height,
	     info.resources.layers, info.resources.levels, static_cast<int>(info.pixel_format),
	     static_cast<uint32_t>(info.tile_mode),
	     m_graphics.presented_frames.load(std::memory_order_relaxed),
	     image.IsGpuModified() ? 'G' : '-', image.IsBufferModified() ? 'B' : '-',
	     image.IsCpuDirty() ? 'C' : '-', event.c_str());
	const auto layers = std::max(info.resources.layers, 1u);
	if (!guest_glyphs || info.IsVolume() || info.data.size % (uint64_t {layers} * 4u) != 0 ||
	    info.data.size > FrameGraph::MaxVerifyBytes || followed.guest_logs >= 64) {
		return;
	}
	std::vector<uint32_t> words(info.data.size / sizeof(uint32_t));
	if (!LibKernel::Memory::TryReadBacking(info.data.address, words.data(), info.data.size)) {
		LOGF("Trace image #%u: its guest bytes cannot be read\n", followed.serial);
		return;
	}
	GuestToHostLayout(words.data(), words.size(), info.GetColorTransform());
	const auto summary = SummarizeLayers(words.data(), words.size(), layers, info.pixel_format, 6);
	auto&      last    = m_trace->guest_glyphs[info.data.address];
	if (last == summary.glyphs) {
		LOGF("Trace image #%u: its guest bytes, a glyph a layer, are as last logged\n",
		     followed.serial);
		return;
	}
	followed.guest_logs++;
	last = summary.glyphs;
	LOGF("Trace image #%u guest bytes, a glyph a layer, padding included (%s):\n", followed.serial,
	     LayerGlyphLegend);
	LogLayerSummary(fmt::format("Trace image #{} guest", followed.serial), summary);
}

// The lines of a followed image's bindings as a target and of the draws into it.
static constexpr uint32_t MaxTargetLogs = 400;

void TextureCache::TraceTargetDraw(ImageId id, const char* kind, const std::string& state) {
	std::scoped_lock lock {m_lock};
	if (m_trace == nullptr) {
		return;
	}
	const auto found = m_trace->images.find(id);
	if (found == m_trace->images.end()) {
		return;
	}
	auto&      followed = found->second;
	const auto frame    = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (followed.draw_frame != frame) {
		followed.draw_frame = frame;
		followed.draw_states.clear();
	}
	if (followed.draw_states.size() >= 8 || followed.target_logs >= MaxTargetLogs ||
	    !followed.draw_states.insert(state).second) {
		return;
	}
	if (++followed.target_logs == MaxTargetLogs) {
		LOGF("Trace image #%u: its target bindings and draws are not logged after these\n",
		     followed.serial);
	}
	TraceImageEvent(id, fmt::format("a draw into it as the {} target: {}", kind, state), false);
}

void TextureCache::TraceTargetBound(ImageId id, const ImageDesc& desc, const char* kind) {
	auto&      followed = m_trace->images[id];
	const auto frame    = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (followed.bind_frame == frame || followed.target_logs >= MaxTargetLogs) {
		return;
	}
	followed.bind_frame = frame;
	if (++followed.target_logs == MaxTargetLogs) {
		LOGF("Trace image #%u: its target bindings and draws are not logged after these\n",
		     followed.serial);
	}
	const auto& meta = desc.info.metadata.range;
	TraceImageEvent(
	    id,
	    fmt::format("bound as the {} target: 0x{:012x}+0x{:x} {}x{} mip {} layer {}{}", kind,
	                desc.info.data.address, desc.info.data.size, desc.info.extent.width,
	                desc.info.extent.height, desc.view_info.base_level, desc.view_info.base_layer,
	                desc.info.HasMetadata()
	                    ? fmt::format("; metadata 0x{:012x}+0x{:x} ({}), its memory: {}",
	                                  meta.address, meta.size,
	                                  m_surface_metas.contains(meta.address) ? "known" : "new",
	                                  LibKernel::Memory::DescribeMappings(meta.address, meta.size))
	                    : std::string("; no metadata")),
	    false);
}

void TextureCache::TraceImageChanged(Image& image, const char* how, bool gpu_write) {
	if (m_trace == nullptr) {
		return;
	}
	const auto found = m_trace->images.find(image.self_id);
	if (found == m_trace->images.end()) {
		return;
	}
	found->second.changed = true;
	found->second
	    .writers[gpu_write && s_trace_shader != 0 ? fmt::format("cs {:016x}", s_trace_shader)
	                                              : std::string(how)]++;
}

void TextureCache::TraceCpuWrite(ImageId id, const Image& image, uint64_t address, uint64_t size) {
	const auto found = m_trace->images.find(id);
	if (found == m_trace->images.end()) {
		return;
	}
	const auto count = ++found->second.cpu_writes;
	if (count > 8 && !std::has_single_bit(count)) {
		return;
	}
	const auto offset = address > image.info.data.address ? address - image.info.data.address : 0;
	const auto layers = std::max(image.info.resources.layers, 1u);
	const auto slice  = image.info.data.size / layers;
	TraceImageEvent(id,
	                fmt::format("CPU write 0x{:012x}+0x{:x} (offset 0x{:x}, layer {}; {} so far){}",
	                            address, size, offset, slice != 0 ? offset / slice : 0, count,
	                            image.IsGpuModified() ? ": its GPU pixels give way to the guest "
	                                                    "bytes at its next use"
	                                                  : ""),
	                false);
}

void TextureCache::TraceFrameEnd() {
	if (m_trace == nullptr && m_nan_check == nullptr) {
		return;
	}
	std::scoped_lock lock {m_lock};
	if (m_nan_check != nullptr) {
		RunNanCheck();
	}
	if (m_gpu_scan != nullptr && m_gpu_scan->scratch != nullptr &&
	    m_graphics.presented_frames.load(std::memory_order_relaxed) >
	        m_gpu_scan->last_frame + GpuScan::IdleFrames) {
		// The scans that used it keep it until they are done.
		m_gpu_scan->scratch.reset();
	}
	if (m_trace == nullptr) {
		return;
	}
	const auto       frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	for (auto it = m_trace->images.begin(); it != m_trace->images.end();) {
		auto& [id, followed] = *it;
		auto* image          = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered) {
			it = m_trace->images.erase(it);
			continue;
		}
		if (followed.changed && ImageDump::Enabled() &&
		    ImageDump::Wanted(image->info.data.address, image->info.extent.width,
		                      image->info.extent.height, image->info.resources.layers,
		                      image->info.IsDepth())) {
			(void)DumpImageLocked(*image, true, 0,
			                      fmt::format("trace_f{}_n{}", frame, followed.serial),
			                      fmt::format("Trace image #{} frame {}", followed.serial, frame));
		}
		if (followed.changed && m_trace->snapshots < m_trace->max_snapshots) {
			std::string writers;
			for (const auto& [who, count]: followed.writers) {
				writers += fmt::format(" {} x{}", who, count);
			}
			auto& history = m_trace->history[image->info.data.address];
			if (history == nullptr) {
				history = std::make_shared<TraceHistory>();
			}
			const auto prefix = fmt::format("Trace image #{} frame {}", followed.serial, frame);
			if (ScanImageLocked(
			        *image, prefix, FrameGraph::MaxVerifyBytes, history,
			        fmt::format("{} 0x{:012x}: mip 0 as the GPU holds it, changed by:{}", prefix,
			                    image->info.data.address, writers))) {
				if (++m_trace->snapshots == m_trace->max_snapshots) {
					LOGF("Trace image: the snapshots stop here (KYTY_TRACE_IMAGES_SNAPSHOTS=%u)\n",
					     m_trace->max_snapshots);
				}
			} else if (!followed.unscanned_logged) {
				followed.unscanned_logged = true;
				LOGF("%s: no snapshot (its format is block-compressed or under a word a texel, or "
				     "mip 0 is over %" PRIu64 " MB); changed by:%s\n",
				     prefix.c_str(), FrameGraph::MaxVerifyBytes >> 20u, writers.c_str());
			}
			followed.changed = false;
			followed.writers.clear();
		}
		++it;
	}
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto& image = m_slot_images[id];
		// A write at the image's base is the same resource seen as a buffer. One elsewhere in or
		// around it leaves an image the GPU drew or stored alone (the buffer does not hold those
		// texels). KYTY_TEXTURE_BUFFER_WRITES=1 lets a texture whose texels are still the guest
		// bytes take such a write too (the buffer holds those bytes and the write). Off by
		// default: GT7's lights, the reason for it, never came from a GPU buffer write (the game
		// hands its lit shaders one or two lights). The device loss on leaving a race that was
		// blamed on it went on without it (EQAA draws shading per sample, shaders.cpp).
		static const bool texture_buffer_writes = [] {
			const char* text = std::getenv("KYTY_TEXTURE_BUFFER_WRITES");
			return text != nullptr && text[0] == '1';
		}();
		const bool guest_texels =
		    texture_buffer_writes && !image.IsGpuModified() && !image.usage.render_target &&
		    !image.usage.depth_target && !image.usage.storage && !image.usage.video_out &&
		    image.info.samples == 1 &&
		    image.info.metadata.compression == VideoOutCompression::Uncompressed &&
		    image.Overlaps(address, size);
		if (image.info.data.address != address && !guest_texels) {
			if (TracesImage(id)) {
				auto&      followed = m_trace->images[id];
				const auto count    = ++followed.inner_writes;
				if (count <= 16 || std::has_single_bit(count)) {
					TraceImageEvent(
					    id,
					    fmt::format("GPU buffer write 0x{:012x}+0x{:x} over it, not at its base "
					                "(offset 0x{:x}; writers {}; {} so far): it keeps its pixels",
					                address, size,
					                address > image.info.data.address
					                    ? address - image.info.data.address
					                    : 0,
					                BufferCache::GpuWritersOf(address, size), count),
					    false);
				}
			}
			continue;
		}
		if (image.info.data.address != address) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				LOGF("TextureCache: GPU buffer write 0x%016" PRIx64 "+0x%" PRIx64
				     " covers texture 0x%016" PRIx64 "+0x%" PRIx64
				     " (%ux%u, %u layers, format %u): the texture takes the write\n",
				     address, size, image.info.data.address, image.info.data.size,
				     image.info.extent.width, image.info.extent.height, image.info.resources.layers,
				     static_cast<uint32_t>(image.info.pixel_format));
			}
		}
		if (TracesImage(id)) {
			TraceImageEvent(
			    id,
			    fmt::format("GPU buffer write 0x{:012x}+0x{:x} at its base (writers {}): "
			                "its next use takes the buffer's bytes",
			                address, size, BufferCache::GpuWritersOf(address, size)),
			    false);
		}
		if (image.IsGpuModified()) {
			static std::atomic<uint64_t> replaced {0};
			const auto                   count = ++replaced;
			if (count <= 32 || std::has_single_bit(count)) {
				// The writer: the shader being bound, or none (a copy, a fill or a DMA).
				const auto writer = BufferCache::s_diag_shader_hash;
				LOGF("TextureCache: GPU buffer write 0x%012" PRIx64 "+0x%" PRIx64
				     " at the base of GPU-written image 0x%012" PRIx64 "+0x%" PRIx64
				     " %ux%u layers %u mips %u fmt %d: its next use takes the buffer's bytes over "
				     "its pixels (%" PRIu64 " so far; written by %s)\n",
				     address, size, image.info.data.address, image.info.data.size,
				     image.info.extent.width, image.info.extent.height, image.info.resources.layers,
				     image.info.resources.levels, static_cast<int>(image.info.pixel_format), count,
				     writer != 0 ? fmt::format("shader {:016x}", writer).c_str()
				                 : "a copy, a fill or a DMA");
			}
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
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
			if (m_trace != nullptr) {
				TraceCpuWrite(id, *owner, address, size);
			}
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

bool TextureCache::IsColorMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	return m_color_metas.contains(address);
}

std::vector<std::pair<uint64_t, uint64_t>> TextureCache::PendingWriteBacks() const {
	std::lock_guard lock(m_pending_writebacks->mutex);
	return m_pending_writebacks->ranges;
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	return (found->second.clear_mask & (1u << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address, std::optional<uint32_t> fill) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	found->second.fill_known = fill.has_value();
	found->second.fill_value = fill.value_or(0);
	return true;
}

uint64_t TextureCache::MetaSize(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end() ? found->second.size : 0;
}

std::optional<uint32_t> TextureCache::MetaFillValue(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || !found->second.fill_known) {
		return std::nullopt;
	}
	return found->second.fill_value;
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
	ForgetLandedReadbacks(address, size);
	ForgetReadbackPrefill(address, size);
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	std::erase_if(m_color_metas,
	              [&](uint64_t base) { return base >= address && base < address + size; });
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id, "its memory was unmapped", true);
	}
}

uint64_t TextureCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

namespace {
// Why the collector does not free: over each TexStats window, candidates it visited and what
// happened to them.
struct GcSkipStats {
	uint64_t visited      = 0;
	uint64_t freed        = 0;
	uint64_t freed_mb     = 0;
	uint64_t gone         = 0; // no longer registered
	uint64_t depth        = 0; // part of a depth/stencil association
	uint64_t gpu_tiled    = 0; // GPU-written tiled: needs a write-back it cannot do
	uint64_t gpu_unpress  = 0; // GPU-written, not under pressure
	uint64_t gpu_download = 0; // GPU-written, write-back failed
	uint64_t young        = 0; // runs where the LRU had nothing old enough
	uint64_t written_back = 0; // GPU-written tiled images written back, then freed
	uint64_t bindless     = 0; // bindless images freed: unsampled in the last usage probe
	uint64_t small_target = 0; // render targets of at most 1024x1024 kept (glyphs, cached layers)
	uint64_t pending      = 0; // evicted, write-back not landed yet
	uint64_t revived      = 0; // used again before the write-back landed: stays
};
GcSkipStats g_gc_skip;
} // namespace

void TextureCache::ReleaseDroppedByGuest(ImageId id) {
	std::scoped_lock lock {m_lock};
	if (auto* image = m_slot_images.try_get(id); image != nullptr) {
		image->frame_guest_dropped = m_graphics.presented_frames.load(std::memory_order_relaxed);
		m_guest_dropped.push_back(id);
	}
}

void TextureCache::RunGarbageCollector() {
	struct GcSection {
		uint64_t begin = StallWatch::NowNs();
		~GcSection() {
			StallWatch::NoteSection(StallWatch::Section::TextureGc, begin, StallWatch::NowNs());
		}
	} gc_section;
	std::scoped_lock lock {m_lock};
	m_gc_tick++;
	const uint64_t clock = LruClock();
	// Evictions whose write-back has landed: free the image now, unless it was used since (then
	// its pixels are newer than the guest bytes and it stays, as if never a candidate).
	std::erase_if(m_pending_evictions, [&](const PendingEviction& pending) {
		if (!pending.landed->load(std::memory_order_acquire)) {
			return false;
		}
		m_evicting.erase(pending.id);
		auto* image = m_slot_images.try_get(pending.id);
		if (image == nullptr || !image->registered || image->info.data.address != pending.address) {
			return true;
		}
		if (image->frame_accessed_last != pending.frame) {
			g_gc_skip.revived++;
			return true;
		}
		const auto freed = image->AccountedSize();
		g_gc_skip.freed++;
		g_gc_skip.freed_mb += freed >> 20u;
		FreeImage(pending.id, "evicted, written back", true);
		return true;
	});
	// KYTY_BISECT's gc:off: nothing else is collected while the step is on.
	if (Bisect::GcOff()) {
		return;
	}
	// Memory pressure (GraphicContext::MemoryShortfall: device use past 90 % of the video memory
	// target): the thresholds sit that far below what the cache holds, so the collector frees its
	// oldest images until there is room again.
	if (m_trigger_gc_memory != m_applied_trigger_gc_memory ||
	    m_pressure_gc_memory != m_applied_pressure_gc_memory ||
	    m_critical_gc_memory != m_applied_critical_gc_memory) {
		m_base_trigger_gc_memory  = m_trigger_gc_memory;
		m_base_pressure_gc_memory = m_pressure_gc_memory;
		m_base_critical_gc_memory = m_critical_gc_memory;
	}
	if (const auto shortfall = m_graphics.MemoryShortfall(); shortfall != 0) {
		const auto collectable = CollectableMemory();
		const auto cap =
		    std::max<uint64_t>(collectable > shortfall ? collectable - shortfall : 0, 512ull << 20u);
		m_trigger_gc_memory  = std::min(m_base_trigger_gc_memory, cap);
		m_pressure_gc_memory = std::min(m_base_pressure_gc_memory, cap);
		m_critical_gc_memory = std::min(m_base_critical_gc_memory, cap + cap / 8);
	} else {
		m_trigger_gc_memory  = m_base_trigger_gc_memory;
		m_pressure_gc_memory = m_base_pressure_gc_memory;
		m_critical_gc_memory = m_base_critical_gc_memory;
	}
	m_applied_trigger_gc_memory  = m_trigger_gc_memory;
	m_applied_pressure_gc_memory = m_pressure_gc_memory;
	m_applied_critical_gc_memory = m_critical_gc_memory;
	if (!m_guest_dropped.empty()) {
		// Engine method: the guest's own streaming dropped these textures from its heaps, as an
		// engine unloads a texture; free them once no draw has used them for a few frames, not
		// when the cache reaches its memory threshold. Images the GPU wrote keep their data for
		// the regular collector (it writes them back); one a heap entry or a render target uses
		// again stays.
		const auto frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
		std::erase_if(m_guest_dropped, [&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr || !image->registered || image->bindless_pinned ||
			    image->binding.is_target || image->IsGpuModified()) {
				return true;
			}
			// A grace of 300 frames (10-20 s): the guest drops entries and takes the same texture
			// back shortly after (the jungle slash stalled 13 s with a grace of 3 frames, each
			// texture freed and created again). The drop time counts as a use.
			if (std::max(image->frame_accessed_last, image->frame_guest_dropped) + 300 > frame) {
				return false; // used or dropped lately: look again later
			}
			m_guest_dropped_freed++;
			m_guest_dropped_freed_bytes += image->AccountedSize();
			FreeImage(id, "dropped by the guest", true);
			return true;
		});
		if (m_gc_tick % 500 == 0) {
			LOGF("TexGuestDrop: frame=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 "MB) waiting=%zu\n",
			     frame, m_guest_dropped_freed, m_guest_dropped_freed_bytes >> 20u,
			     m_guest_dropped.size());
		}
	}
	if (m_graphics.CanReportMemoryUsage()) {
		// Pressure is judged by this cache's own image bytes, which RegisterImage and
		// UnregisterImage already track. Device-wide usage counts the images this cache has
		// spilled to host memory (and every buffer), so it reads "critical" no matter how
		// small the cache is -- and never falls, so the collector cannot tell that it is the
		// one holding too much. Spider-Man's world then grew to 7.6 GB of images on a 6 GB
		// card, all of it spilled, and every shader access crossed PCIe (3 fps).
		// KYTY_GC_DEVICE_BYTES=1 restores the old policy, as in the buffer cache.
		static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
		if (device_bytes) {
			m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
		}
	}
	if (m_pinned_frame != clock) {
		// Heap textures follow the guest (settled when it writes an entry, freed when it drops
		// them) and the collector cannot free them; counted against the bound they kept the cache
		// over it, and it freed every other image again and again (33,916 in 4 minutes of a new
		// game, the frame rate a fifth lower). The bounds apply to the images it can free.
		m_pinned_frame = clock;
		m_pinned_bytes = 0;
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered && image.bindless_pinned) {
				m_pinned_bytes += image.AccountedSize();
			}
		});
	}
	if (CollectableMemory() < m_trigger_gc_memory) {
		return;
	}
	// The collector runs once per guest submission -- this title submits hundreds of times a
	// frame -- so a budget counted per run is really that many evictions per frame, and every
	// eviction is an image recreated, synchronized and submitted. Budget the work per frame
	// instead: 7 600 submissions a frame came from a per-run budget of 256.
	if (m_gc_budget_frame != clock) {
		m_gc_budget_frame       = clock;
		m_gc_freed_bytes_frame  = 0;
		m_gc_freed_images_frame = 0;
		m_gc_written_back_bytes_frame = 0;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool pressured  = CollectableMemory() >= m_pressure_gc_memory;
		bool aggressive = allow_aggressive && CollectableMemory() >= m_critical_gc_memory;
		// Ages in frames. This collector runs once per guest submission, and a title can
		// submit a hundred times a frame: an age counted in its ticks evicted textures inside
		// the frame that used them, to recreate them the next (11 000 images a second at the
		// title menu). An image used in the frame being recorded is never a candidate, at any
		// pressure; over budget the collector reaches back only as far as the previous frame.
		const uint64_t age = std::min<uint64_t>(aggressive ? 1 : pressured ? 4 : 16, clock);
		// Under pressure the work is budgeted per frame, in bytes actually freed, and in
		// proportion to the excess, so the cache converges on its threshold instead of either
		// churning (a per-run budget: 7 600 submissions a frame) or growing without bound (a
		// small fixed count, spent on GPU-written render targets that are skipped: 13.5 GB).
		// Only frees count against it. A cap on images per frame bounds the synchronization
		// that recreating them costs.
		constexpr uint64_t MiB             = 1024 * 1024;
		constexpr size_t   MaxFreesInFrame = 1024;
		uint64_t           byte_budget     = 0;
		size_t             deletions       = 10;
		if (pressured || aggressive) {
			const auto threshold = aggressive ? m_critical_gc_memory : m_pressure_gc_memory;
			const auto excess    = CollectableMemory() - threshold;
			byte_budget = aggressive ? std::max<uint64_t>(64 * MiB, excess / 4)
			                         : std::max<uint64_t>(16 * MiB, excess / 8);
			if (m_gc_freed_bytes_frame >= byte_budget ||
			    m_gc_freed_images_frame >= MaxFreesInFrame) {
				return;
			}
			deletions = MaxFreesInFrame - m_gc_freed_images_frame;
		}
		// Skipped candidates (GPU-written tiled images, depth associations) do not use up the
		// budget, so look further down the LRU than the number of frees wanted.
		const size_t         visit_limit = pressured || aggressive ? deletions * 4 : deletions;
		std::vector<ImageId> candidates;
		candidates.reserve(std::min<size_t>(visit_limit, 4096));
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(clock - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() >= visit_limit || candidates.size() >= 4096;
		});
		if (candidates.empty() && (pressured || aggressive)) {
			g_gc_skip.young++;
		}
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			if (!(pressured || aggressive)) {
				--deletions;
			}
			g_gc_skip.visited++;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered) {
				g_gc_skip.gone++;
				continue;
			}
			if (owner->depth_id) {
				g_gc_skip.depth++;
				continue;
			}
			if (m_evicting.contains(id)) {
				g_gc_skip.pending++;
				continue;
			}
			// Small render targets are the layers the game draws once and samples from then on
			// (glyph atlases, cached UI): a few MB each, and a lost one is lost text (letters went
			// missing when they were evicted under "critical", 2026-10-09). They are never evicted;
			// the guest dropping them (ReleaseDroppedByGuest) is what frees them.
			if (owner->usage.render_target && owner->info.extent.width <= 1024 &&
			    owner->info.extent.height <= 1024) {
				g_gc_skip.small_target++;
				continue;
			}
			if (owner->bindless_pinned) {
				// Pinned for bindless draws; under pressure one no draw sampled during the last
				// usage probe may go (ISSUES #15: on 4-8 GB cards these grew to 2-5 GB and never
				// shrank).
				if (!(pressured || aggressive) || !m_bindless_evictable.contains(id)) {
					continue;
				}
				m_bindless_evictable.erase(id);
				g_gc_skip.bindless++;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				// A GPU-written tiled image was never evicted: render targets and textures the
				// GPU decompresses are almost all of a 4K title's images, so under pressure the
				// collector found nothing else it could free (Spider-Man: 3 500 of every 4 000
				// candidates) and the cache grew to 13.5 GB until the driver crashed in
				// present. Under pressure such an image is now written back, detiled, like any
				// other, within a per-frame write-back budget; candidates are never ones used in
				// the frame being recorded. KYTY_GC_KEEP_TILED=1 restores the old rule.
				static const bool keep_tiled = std::getenv("KYTY_GC_KEEP_TILED") != nullptr;
				constexpr uint64_t WriteBackPerFrame = 256 * MiB;
				if (safe && owner->info.IsTiled() &&
				    (keep_tiled || !(pressured || aggressive) ||
				     (m_gc_written_back_bytes_frame != 0 &&
				      m_gc_written_back_bytes_frame + owner->info.data.size > WriteBackPerFrame))) {
					g_gc_skip.gpu_tiled++;
					continue;
				}
				if (safe && !pressured) {
					g_gc_skip.gpu_unpress++;
					continue;
				}
				if (!safe) {
					// GPU-written, and CPU- or copy-written as well: its contents exist only in
					// this image and cannot be written back as they are. Freeing it lost them:
					// under a 6 GiB bound the Leap Attack prompt read "L p Att k" (the glyph atlas,
					// filled by both). It stays until it is safe or the guest drops it.
					g_gc_skip.gpu_download++;
					continue;
				}
				std::shared_ptr<std::atomic<bool>> landed;
				if (safe && !DownloadImageMemory(id, true, &landed)) {
					g_gc_skip.gpu_download++;
					continue;
				}
				if (safe && owner->info.IsTiled()) {
					m_gc_written_back_bytes_frame += owner->info.data.size;
					g_gc_skip.written_back++;
				}
				// Freed once the write-back has landed (the top of the next run); the bytes count
				// against this frame's budget now, since they are going.
				m_pending_evictions.push_back(
				    {id, owner->info.data.address, owner->frame_accessed_last, std::move(landed)});
				m_evicting.insert(id);
				if (pressured || aggressive) {
					--deletions;
					m_gc_freed_images_frame++;
					m_gc_freed_bytes_frame += owner->AccountedSize();
					if (m_gc_freed_bytes_frame >= byte_budget) {
						break;
					}
				}
				continue;
			}
			const auto freed = owner->AccountedSize();
			g_gc_skip.freed++;
			g_gc_skip.freed_mb += freed >> 20u;
			FreeImage(id, "evicted", true);
			if (pressured || aggressive) {
				--deletions;
				m_gc_freed_images_frame++;
				m_gc_freed_bytes_frame += freed;
				if (m_gc_freed_bytes_frame >= byte_budget) {
					break;
				}
			}
			if (CollectableMemory() < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (CollectableMemory() < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (CollectableMemory() >= m_critical_gc_memory) {
		collect(true);
	}
	if (m_gc_tick % 500 == 0) {
		uint64_t images = 0;
		uint64_t bytes  = 0;
		// The render targets of at most 1024x1024 the collector keeps (small_target below).
		uint64_t small_targets      = 0;
		uint64_t small_target_bytes = 0;
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered) {
				images++;
				bytes += image.AccountedSize();
				if (image.usage.render_target && image.info.extent.width <= 1024 &&
				    image.info.extent.height <= 1024) {
					small_targets++;
					small_target_bytes += image.AccountedSize();
				}
			}
		});
		LOGF("TexStats: runs=%" PRIu64 " frame=%" PRIu64 " images=%" PRIu64 " image_bytes=%" PRIu64
		     "MB device_used=%" PRIu64 "MB pressure=%" PRIu64 "MB critical=%" PRIu64
		     "MB small_targets=%" PRIu64 " (%" PRIu64 "MB)\n",
		     m_gc_tick, m_graphics.presented_frames.load(std::memory_order_relaxed), images,
		     bytes >> 20u, m_total_used_memory >> 20u, m_pressure_gc_memory >> 20u,
		     m_critical_gc_memory >> 20u, small_targets, small_target_bytes >> 20u);
		LOGF("TexGc: visited=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 "MB) written_back=%" PRIu64
		     " gone=%" PRIu64 " depth=%" PRIu64 " gpu_tiled=%" PRIu64 " gpu_unpressured=%" PRIu64
		     " gpu_download_failed=%" PRIu64 " nothing_old_enough=%" PRIu64 " bindless=%" PRIu64
		     " small_target=%" PRIu64 " pending=%" PRIu64 " revived=%" PRIu64 "\n",
		     g_gc_skip.visited, g_gc_skip.freed, g_gc_skip.freed_mb, g_gc_skip.written_back,
		     g_gc_skip.gone, g_gc_skip.depth, g_gc_skip.gpu_tiled, g_gc_skip.gpu_unpress,
		     g_gc_skip.gpu_download, g_gc_skip.young, g_gc_skip.bindless, g_gc_skip.small_target,
		     g_gc_skip.pending, g_gc_skip.revived);
		g_gc_skip = {};
	}
}

void TextureCache::SetBindlessEvictable(std::span<const ImageId> ids) {
	std::scoped_lock lock {m_lock};
	m_bindless_evictable.clear();
	m_bindless_evictable.insert(ids.begin(), ids.end());
}

bool TextureCache::ReadbacksOnTheirWay() {
	{
		std::scoped_lock lock {m_lock};
		if (!m_download_images.empty()) {
			return true;
		}
	}
	std::lock_guard lock(m_pending_writebacks->mutex);
	return !m_pending_writebacks->ranges.empty();
}

namespace {

// KYTY_MS_DEPTH_SAMPLE0=0: single-sample reads of multisample depth and stencil make no copy.
// Since round 124 the shader's image is a single-sample one there, and a multisample image bound
// to it is a mismatch Vulkan leaves undefined, so those reads get a null texture: for comparisons
// only.
bool MsDepthSample0Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MS_DEPTH_SAMPLE0");
		const bool  on    = value == nullptr || value[0] != '0';
		LOGF("Multisample depth read as single-sample: %s (KYTY_MS_DEPTH_SAMPLE0)\n",
		     on ? "from a copy of its first sample" : "a null texture (no copy)");
		return on;
	}();
	return enabled;
}

// The reads and copies of SingleSampleDepth and SingleSampleStencil, logged every 10 s.
struct SingleSampleStats {
	uint64_t                              depth_reads    = 0;
	uint64_t                              depth_copies   = 0;
	uint64_t                              stencil_reads  = 0;
	uint64_t                              stencil_copies = 0;
	std::chrono::steady_clock::time_point since          = std::chrono::steady_clock::now();
};
SingleSampleStats g_single_sample_stats;

void LogSingleSampleStats(size_t depth_images, size_t stencil_images) {
	auto&      stats = g_single_sample_stats;
	const auto now   = std::chrono::steady_clock::now();
	if (now - stats.since < std::chrono::seconds(10)) {
		return;
	}
	LOGF("Multisample depth read as single-sample (%.0f s): %" PRIu64 " depth reads, %" PRIu64
	     " copies of the first sample (%zu depth images); %" PRIu64 " stencil reads, %" PRIu64
	     " copies (%zu stencil planes)\n",
	     std::chrono::duration<double>(now - stats.since).count(), stats.depth_reads,
	     stats.depth_copies, depth_images, stats.stencil_reads, stats.stencil_copies,
	     stencil_images);
	stats = {};
}

} // namespace

ImageId TextureCache::SingleSampleDepth(ImageId id) {
	if (!MsDepthSample0Enabled()) {
		return id;
	}
	std::scoped_lock lock {m_lock};
	const auto*      source = m_slot_images.try_get(id);
	if (source == nullptr || source->backing.samples <= 1 || !source->info.IsDepth() ||
	    source->backing.image == nullptr || source->backing.image_type != vk::ImageType::e2D ||
	    source->info.resources.layers == 0 ||
	    source->info.resources.layers != source->backing.layers ||
	    source->info.resources.levels != 1 || source->depth_id) {
		return id;
	}
	// What FindTexture does with an image it binds, done here for the source the copy is made
	// from: its use is noted, and what the CPU or a buffer write put in its memory is uploaded
	// first (which changes its content serial, so the copy is made again).
	TouchImage(m_slot_images[id]);
	if (source->registered && !source->binding.needs_rebind) {
		RefreshImage(id);
		source = &m_slot_images[id];
	}
	const auto width  = source->info.extent.width;
	const auto height = source->info.extent.height;
	const auto layers = source->info.resources.layers;
	auto&      entry  = m_single_sample_depths[id];
	auto*      copy   = entry.copy ? m_slot_images.try_get(entry.copy) : nullptr;
	if (copy == nullptr || copy->info.extent.width != width || copy->info.extent.height != height ||
	    copy->info.resources.layers != layers) {
		if (copy != nullptr) {
			const auto old = entry.copy;
			m_scheduler.DeferRelease([this, old] { ReleaseImage(old); });
		}
		ImageInfo info {};
		info.pixel_format    = vk::Format::eD32Sfloat;
		info.guest_format    = source->info.guest_format;
		info.type            = source->info.type;
		info.extent          = {width, height, 1};
		info.resources       = {1, layers};
		info.pitch           = width;
		info.bytes_per_block = 4;
		info.samples         = 1;
		info.tile_mode       = Prospero::TileMode::kLinear;
		info.mip_layout[0]   = {0, uint64_t {width} * height * 4u, width, height};
		const auto guest     = source->info.data.address;
		const auto samples   = source->backing.samples;
		entry.copy           = InsertImage(info);
		entry.serial         = UINT64_MAX;
		copy                 = &m_slot_images[entry.copy];
		source               = &m_slot_images[id];
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Multisample depth read as single-sample: 0x%012" PRIx64 " %ux%u, %u layers, %u "
			     "samples: read from a copy of its first sample\n",
			     guest, width, height, layers, samples);
		}
	}
	copy->tick_accessed_last = m_scheduler.CurrentTick();
	g_single_sample_stats.depth_reads++;
	if (entry.serial != source->content_serial) {
		m_blit_helper.CopyMsDepthSample0(m_slot_images[id], *copy);
		entry.serial = m_slot_images[id].content_serial;
		g_single_sample_stats.depth_copies++;
	}
	LogSingleSampleStats(m_single_sample_depths.size(), m_single_sample_stencils.size());
	return entry.copy;
}

ImageId TextureCache::SingleSampleStencil(ImageId depth_id) {
	if (!MsDepthSample0Enabled()) {
		return depth_id;
	}
	std::scoped_lock lock {m_lock};
	const auto*      source = m_slot_images.try_get(depth_id);
	if (source == nullptr || source->backing.samples <= 1 || !source->info.IsDepth() ||
	    !HasStencilAspect(source->backing.format) || source->backing.image == nullptr ||
	    source->backing.image_type != vk::ImageType::e2D || source->info.resources.layers == 0 ||
	    source->info.resources.layers != source->backing.layers ||
	    source->info.resources.levels != 1 || source->depth_id) {
		return depth_id;
	}
	// As SingleSampleDepth (FindTexture's refresh), for the depth image and its stencil planes.
	TouchImage(m_slot_images[depth_id]);
	if (source->registered && !source->binding.needs_rebind) {
		RefreshImage(depth_id);
		const auto stencil_ids = m_slot_images[depth_id].stencil_ids;
		for (const auto stencil_id: stencil_ids) {
			if (const auto* stencil = m_slot_images.try_get(stencil_id);
			    stencil != nullptr && stencil->depth_id == depth_id) {
				RefreshImage(stencil_id);
			}
		}
		source = &m_slot_images[depth_id];
	}
	// What the stencil holds changes with the depth image (its draws, clears and copies) and with
	// its stencil planes (their own clears and uploads).
	const auto serial_of = [&](const Image& depth) {
		uint64_t serial = depth.content_serial;
		for (const auto stencil_id: depth.stencil_ids) {
			if (const auto* stencil = m_slot_images.try_get(stencil_id); stencil != nullptr) {
				serial = serial * 31u + stencil->content_serial + 1u;
			}
		}
		return serial;
	};
	const auto width  = source->info.extent.width;
	const auto height = source->info.extent.height;
	const auto layers = source->info.resources.layers;
	auto&      entry  = m_single_sample_stencils[depth_id];
	auto*      copy   = entry.copy ? m_slot_images.try_get(entry.copy) : nullptr;
	if (copy == nullptr || copy->info.extent.width != width || copy->info.extent.height != height ||
	    copy->info.resources.layers != layers) {
		if (copy != nullptr) {
			const auto old = entry.copy;
			m_scheduler.DeferRelease([this, old] { ReleaseImage(old); });
		}
		ImageInfo info {};
		info.pixel_format    = vk::Format::eR8Uint;
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.type            = source->info.type;
		info.extent          = {width, height, 1};
		info.resources       = {1, layers};
		info.pitch           = width;
		info.bytes_per_block = 1;
		info.samples         = 1;
		info.tile_mode       = Prospero::TileMode::kLinear;
		info.mip_layout[0]   = {0, uint64_t {width} * height, width, height};
		const auto guest     = source->info.data.address;
		const auto samples   = source->backing.samples;
		entry.copy           = InsertImage(info);
		entry.serial         = UINT64_MAX;
		copy                 = &m_slot_images[entry.copy];
		source               = &m_slot_images[depth_id];
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Multisample depth read as single-sample: the stencil of 0x%012" PRIx64
			     " %ux%u, %u layers, %u samples: read from a copy of its first sample\n",
			     guest, width, height, layers, samples);
		}
	}
	copy->tick_accessed_last = m_scheduler.CurrentTick();
	g_single_sample_stats.stencil_reads++;
	if (const auto serial = serial_of(*source); entry.serial != serial) {
		m_blit_helper.CopyMsStencilSample0(m_slot_images[depth_id], *copy);
		entry.serial = serial;
		g_single_sample_stats.stencil_copies++;
	}
	LogSingleSampleStats(m_single_sample_depths.size(), m_single_sample_stencils.size());
	return entry.copy;
}

ImageId TextureCache::SubExtentCopy(ImageId id, const ImageInfo& requested) {
	std::scoped_lock lock {m_lock};
	const auto*      source = m_slot_images.try_get(id);
	if (source == nullptr || !source->registered || source->depth_id ||
	    source->backing.image == nullptr || source->binding.needs_rebind) {
		return id;
	}
	const auto& info = source->info;
	if (info.data.Empty() || info.IsDepth() || info.HasStencil() || info.IsBlock() ||
	    info.IsVolume() || source->backing.samples != 1 ||
	    source->backing.image_type != vk::ImageType::e2D || requested.samples != 1 ||
	    requested.type != info.type || requested.data.address != info.data.address ||
	    requested.data.size != info.data.size || requested.tile_mode != info.tile_mode ||
	    requested.resources != info.resources || requested.resources.levels != 1 ||
	    requested.extent.depth != info.extent.depth || requested.extent.width == 0 ||
	    requested.extent.height == 0 || requested.extent.width > info.extent.width ||
	    requested.extent.height > info.extent.height ||
	    (requested.extent.width == info.extent.width &&
	     requested.extent.height == info.extent.height) ||
	    !ImageViewOps::FormatsCompatible(info.pixel_format, requested.pixel_format)) {
		return id;
	}
	// FindTexture's refresh of an image it binds, for the image the copy is made from (an upload
	// changes its content serial, so the copy is made again).
	TouchImage(m_slot_images[id]);
	RefreshImage(id);
	source            = &m_slot_images[id];
	const auto width  = requested.extent.width;
	const auto height = requested.extent.height;
	const auto tick   = m_scheduler.CurrentTick();
	auto&      copies = m_sub_extent_copies[id];
	size_t     index  = copies.size();
	for (size_t i = 0; i < copies.size(); i++) {
		if (copies[i].width == width && copies[i].height == height &&
		    m_slot_images.try_get(copies[i].copy) != nullptr) {
			index = i;
			break;
		}
	}
	if (index == copies.size()) {
		// At most 4 extents an image: the one used longest ago goes.
		if (copies.size() >= 4) {
			size_t oldest = 0;
			for (size_t i = 1; i < copies.size(); i++) {
				if (copies[i].used < copies[oldest].used) {
					oldest = i;
				}
			}
			if (const auto old = copies[oldest].copy; old) {
				m_scheduler.DeferRelease([this, old] { ReleaseImage(old); });
			}
			copies.erase(copies.begin() + static_cast<std::ptrdiff_t>(oldest));
		}
		ImageInfo copy_info {};
		copy_info.pixel_format    = source->info.pixel_format;
		copy_info.guest_format    = source->info.guest_format;
		copy_info.type            = source->info.type;
		copy_info.extent          = {width, height, source->info.extent.depth};
		copy_info.resources       = source->info.resources;
		copy_info.pitch           = width;
		copy_info.bytes_per_block = source->info.bytes_per_block;
		copy_info.samples         = 1;
		copy_info.tile_mode       = Prospero::TileMode::kLinear;
		copy_info.bgra16          = source->info.bgra16;
		copy_info.mip_layout[0]   = {0, uint64_t {width} * height * source->info.bytes_per_block,
		                             width, height};
		const auto guest          = source->info.data.address;
		const auto source_width   = source->info.extent.width;
		const auto source_height  = source->info.extent.height;
		const auto copy_id        = InsertImage(copy_info);
		copies.push_back({copy_id, width, height, UINT64_MAX, tick});
		index  = copies.size() - 1;
		source = &m_slot_images[id];
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("TextureCache: 0x%012" PRIx64 " %ux%u read as a %ux%u texture (same memory and "
			     "tiling): from a copy of that corner (round 124)\n",
			     guest, source_width, source_height, width, height);
		}
	}
	auto& entry             = copies[index];
	auto& copy              = m_slot_images[entry.copy];
	entry.used              = tick;
	copy.tick_accessed_last = tick;
	if (entry.serial != m_slot_images[id].content_serial) {
		// Within both images: the corner of the source the copy's extent covers.
		copy.CopyImage(m_slot_images[id]);
		entry.serial = m_slot_images[id].content_serial;
	}
	return entry.copy;
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
