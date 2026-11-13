#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/guestWatch.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/gpuPhases.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/frameGraph.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/imageDump.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/tableBindings.h"
#include "graphics/host_gpu/renderer/tableMode.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

struct WatchedReread {
	uint64_t frame   = 0;
	uint64_t address = 0;
	uint64_t size    = 0;
};

std::mutex                 g_watched_reread_mutex;
std::vector<WatchedReread> g_watched_rereads;

void LogWords(const uint32_t* words, size_t count) {
	for (size_t row = 0; row < count; row += 8) {
		std::string line;
		for (size_t k = row; k < std::min(count, row + 8); k++) {
			line += fmt::format(" {:08x}", words[k]);
		}
		LOGF("    +0x%03zx:%s\n", row * 4, line.c_str());
	}
}

} // namespace

// KYTY_WATCH_SHADER=<hash>[,<hash>...] (hex): the bindings of these shaders are logged
// (PrepareBindings); the first also names the shader KYTY_WATCH_SHADER_FRAME and KYTY_DUMP_GUEST's
// relative frames count from.
static const std::vector<uint64_t>& WatchedShaderHashes() {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_WATCH_SHADER");
		while (value != nullptr && *value != '\0') {
			char*      end  = nullptr;
			const auto hash = std::strtoull(value, &end, 16);
			if (end == value) break;
			if (hash != 0) result.push_back(hash);
			value = *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	return hashes;
}

static uint64_t WatchedShaderHash() {
	const auto& hashes = WatchedShaderHashes();
	return hashes.empty() ? 0ull : hashes.front();
}

// The how-manieth use of a watched shader this is (UINT32_MAX: not watched).
static uint32_t WatchedShaderUse(uint64_t hash) {
	const auto& hashes = WatchedShaderHashes();
	if (hash == 0 || std::ranges::find(hashes, hash) == hashes.end()) {
		return UINT32_MAX;
	}
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint32_t> uses;
	const std::lock_guard                         lock(mutex);
	return uses[hash]++;
}

static std::atomic<uint64_t> g_watched_first_frame {UINT64_MAX};

uint64_t WatchedShaderFirstFrame() {
	return g_watched_first_frame.load();
}

bool InWatchedShaderFrame(uint64_t hash, uint64_t presented_frame, std::atomic<uint32_t>& logged) {
	static const uint64_t delta = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER_FRAME");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : UINT64_MAX;
	}();
	if (hash == 0 || hash != WatchedShaderHash()) {
		return false;
	}
	uint64_t unset = UINT64_MAX;
	g_watched_first_frame.compare_exchange_strong(unset, presented_frame);
	return delta != UINT64_MAX && presented_frame == g_watched_first_frame.load() + delta &&
	       logged.fetch_add(1) < 64u;
}

void LogWatchedRereads(BufferCache& cache, uint64_t presented_frames) {
	std::vector<WatchedReread> due;
	{
		std::lock_guard lock(g_watched_reread_mutex);
		for (auto it = g_watched_rereads.begin(); it != g_watched_rereads.end();) {
			if (it->frame <= presented_frames) {
				due.push_back(*it);
				it = g_watched_rereads.erase(it);
			} else {
				++it;
			}
		}
	}
	for (const auto& reread: due) {
		std::vector<uint32_t> words(reread.size / 4u);
		const bool gpu_written = cache.HasGpuDirtyBytes(reread.address, reread.size);
		const bool read =
		    Libs::LibKernel::Memory::TryReadBacking(reread.address, words.data(), reread.size);
		LOGF("WatchShader reread at frame %" PRIu64 ": 0x%012" PRIx64 "+0x%" PRIx64
		     " gpu_written=%d%s\n",
		     presented_frames, reread.address, reread.size, gpu_written ? 1 : 0,
		     read ? "" : " (unreadable)");
		if (read) {
			LogWords(words.data(), words.size());
		}
	}
}

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() ? 1u : static_cast<uint32_t>(binding.resources.size());
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static const char* ShaderStageResourceName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "Vertex";
		case ShaderType::Mesh: return "Mesh";
		case ShaderType::Local: return "Local";
		case ShaderType::TessellationControl: return "Hull";
		case ShaderType::TessellationEvaluation: return "Domain";
		case ShaderType::Pixel: return "Pixel";
		case ShaderType::Compute: return "Compute";
		default: return "Unknown";
	}
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

// A/B switch for bounded buffer writes (KYTY_BOUNDED_BUFFER_WRITES=0 marks every written buffer's
// whole range GPU-written, as before).
static const bool g_bounded_buffer_writes = [] {
	const char* value = std::getenv("KYTY_BOUNDED_BUFFER_WRITES");
	return value == nullptr || value[0] != '0';
}();

// Diagnostics: names the shader whose bindings are prepared, for KYTY_WATCH_GPU_WRITE.
struct DiagShaderScope {
	explicit DiagShaderScope(uint64_t hash) { BufferCache::s_diag_shader_hash = hash; }
	~DiagShaderScope() { BufferCache::s_diag_shader_hash = 0; }
	KYTY_CLASS_NO_COPY(DiagShaderScope);
};

// KYTY_WRITE_BITMAP_MIN_KB (default 1024), as BufferCache's.
static uint64_t WriteBitmapMinBytesForBinding() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_WRITE_BITMAP_MIN_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {1024}) * 1024u;
	}();
	return bytes;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource&    resource,
                    const ShaderRecompiler::IR::BufferWriteExtent* extent, uint32_t& buffer_offset,
                    bool track_writes, uint64_t& tracked_base) {
	buffer_offset = 0;
	tracked_base  = 0;

	const auto& [address, size, id] = source;
	if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	// The stores' addresses bound the bytes this draw or dispatch writes: only those become
	// GPU-written, so the guest's accesses elsewhere in the range do not drain the GPU.
	const bool bounded_write = resource.written && extent != nullptr && extent->valid &&
	                           g_bounded_buffer_writes;
	const auto written_begin = bounded_write ? std::min(extent->begin, size) : 0u;
	const auto written_end   = bounded_write ? std::min(extent->end, size) : size;
	// Unbounded writes through a large range: the shader marks the pages it writes in the GPU
	// write bitmap, and only those are downloaded (BufferCache::ObtainBufferTracked).
	const bool tracked = track_writes && resource.written && !bounded_write &&
	                     size >= WriteBitmapMinBytesForBinding() &&
	                     address + size <= (uint64_t {1} << 40u) &&
	                     context.GetBufferCache().WriteBitmapAddress() != 0;
	auto [buffer, offset] =
	    bounded_write
	        ? context.GetBufferCache().ObtainBufferWritten(address, size, address + written_begin,
	                                                       written_end - std::min(written_begin,
	                                                                              written_end),
	                                                       id)
	    : tracked ? context.GetBufferCache().ObtainBufferTracked(address, size, id)
	              : context.GetBufferCache().ObtainBuffer(address, size, resource.written,
	                                                      resource.formatted, id);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	if (tracked) {
		tracked_base = address - adjustment; // the guest address of the bound range's first byte
	}
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written && written_end > written_begin) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address + written_begin,
		                                                  written_end - written_begin);
	}
	return result;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full ||
	    (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type()))) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	// Dword 5: MAX_MIP and PERF_MOD, a hint to the texture unit that changes nothing a storage
	// image holds. Round 126: any PERF_MOD (before, only 7): T#s with PERF_MOD 0, their upper
	// dwords zero, ended the emulator in a compute dispatch of two testers' races (issues 9 and
	// 13).
	constexpr uint32_t field5_hint_mask = 0x007000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return common && descriptor.fields[4] == expected_field4 &&
	       (descriptor.fields[5] & ~field5_hint_mask) == 0;
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	const bool raw_atomic_storage = resource.atomic && uint_resource &&
	                                (format == Prospero::BufferFormat::k32UInt ||
	                                 format == Prospero::BufferFormat::k32SInt ||
	                                 format == Prospero::BufferFormat::k32Float);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic || raw_atomic_storage ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	// The view must match the dimension the shader declares (Vulkan requires the view type to
	// match the OpTypeImage), so 1D and 3D slots get 1D and 3D null images.
	auto image_type = Prospero::ImageType::kColor2D;
	auto view_type  = vk::ImageViewType::e2D;
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			image_type = Prospero::ImageType::kColor1D;
			view_type  = vk::ImageViewType::e1D;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			image_type = Prospero::ImageType::kColor1D;
			view_type  = vk::ImageViewType::e1DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view_type = vk::ImageViewType::e2DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			image_type = Prospero::ImageType::kColor3D;
			view_type  = vk::ImageViewType::e3D;
			break;
		default: break;
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = image_type;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = view_type;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	const auto texel_shift = info.IsBlock() ? 2u : 0u;
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width >> texel_shift,
		    padded[level].height >> texel_shift,
		};
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || surface_format.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(const TileSurfaceDescription& description, bool metadata,
                                   uint32_t view_levels, uint32_t& levels, uint32_t& base_level) {
	TileSurfaceLayout physical {};
	TileSurfaceLayout view {};
	auto              view_description = description;
	view_description.levels            = levels;
	if (!TileGetTiledTextureLayout(description, physical) ||
	    !TileGetTiledTextureLayout(view_description, view)) {
		return false;
	}
	if (physical.first_tail_level == view.first_tail_level &&
	    physical.block_slice_size == view.block_slice_size &&
	    physical.total_size == view.total_size &&
	    std::equal(std::begin(physical.mips), std::begin(physical.mips) + description.levels,
	               std::begin(view.mips))) {
		return true;
	}
	if (metadata || ((description.layers > 1 || description.depth > 1) &&
	                 physical.block_slice_size != view.block_slice_size)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= description.levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (physical.mips[target] != view.mips[source] ||
			    (target >= physical.first_tail_level) != (source >= view.first_tail_level) ||
			    std::max(description.width >> target, 1u) != std::max(description.width >> source, 1u) ||
			    std::max(description.height >> target, 1u) != std::max(description.height >> source, 1u) ||
			    std::max(description.depth >> target, 1u) != std::max(description.depth >> source, 1u)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			levels     = description.levels;
			base_level = base;
			return true;
		}
	}
	return false;
}

namespace {

// ResolveTexture's description of a texture (tiling, sizes, mip layout, view) depends only on the
// T# and on how the shader uses the image: it was computed again for every bound image of every
// draw, 2.4 % of the GPU thread at the jungle prompt (DEBUGGING.md, 2026-10-05). The texture
// cache lookup that follows still runs every time. KYTY_TEXTURE_DESC_CACHE=0 turns this off.
struct TextureDescKey {
	std::array<uint32_t, 8> dwords {};
	uint32_t                resource_class = 0;
	uint32_t                numeric_class  = 0;
	uint32_t                dimension      = 0;
	uint32_t                mip_mode       = 0;
	uint32_t                mip_count      = 0;
	uint32_t                conversion     = 0;
	uint32_t                swizzle        = 0;
	uint32_t                flags          = 0;

	bool operator==(const TextureDescKey&) const = default;
};
static_assert(sizeof(TextureDescKey) == 64);

struct TextureDescKeyHash {
	size_t operator()(const TextureDescKey& key) const noexcept {
		return static_cast<size_t>(XXH3_64bits(&key, sizeof(key)));
	}
};

// The texture memo: what the texture cache gave a descriptor's binding the last time (the image
// FindImage found, the view base it set, and the view FindTexture made), in the image set it was
// given in (TextureCache::ImageSetGeneration). 88 % of a race frame's 31,600 texture bindings are
// one the command before bound too (1,300 distinct ones): while no image is registered or
// unregistered and the image stays unchanged (TextureCache::ReuseTextureBinding), the binding is
// that again, without FindImage and FindTexture (each a lock and a search).
struct TextureMemo {
	ImageId       id;
	vk::ImageView view       = nullptr;
	uint64_t      generation = 0; // 0: none
	uint32_t      base_level = 0;
	uint32_t      base_layer = 0;
	// The stamp of the descriptor's memory when it was kept (TextureCache::RangeStamp): it is
	// current while the stamp stays (round 78), or, with KYTY_BISECT's texmemo:global, while the
	// generation of the whole image set does, as before.
	uint64_t stamp = 0;
};

struct TextureDescEntry {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint32_t                size              = 0;
	bool                    shader_conversion = false;
	TextureMemo             memo;
};

// The GPU thread's descriptions by descriptor (KYTY_TEXTURE_DESC_CACHE), with their memos.
std::unordered_map<TextureDescKey, TextureDescEntry, TextureDescKeyHash>& TextureDescCache() {
	thread_local std::unordered_map<TextureDescKey, TextureDescEntry, TextureDescKeyHash> cache;
	return cache;
}

// Round 156: a direct-mapped front for the finds in TextureDescCache, two a texture binding (its
// resolve and its memo's record): the map's bucket walk was ~1.4 % of the GPU thread in a race.
// An entry's address stays while the map holds it (a node map); the front is dropped with the
// map's clear (its generation).
struct TextureDescFrontSlot {
	TextureDescKey    key;
	TextureDescEntry* entry      = nullptr;
	uint64_t          generation = 0;
};
struct TextureDescFront {
	static constexpr size_t                 Slots = 1024;
	std::array<TextureDescFrontSlot, Slots> slots {};
	uint64_t                                generation = 1;
};

TextureDescFront& GetTextureDescFront() {
	thread_local const auto front = std::make_unique<TextureDescFront>();
	return *front;
}

// The cache's entry for `key`, or nullptr.
TextureDescEntry* FindTextureDesc(const TextureDescKey& key) {
	auto& cache = TextureDescCache();
	if (Bisect::Cuts156Off()) {
		const auto it = cache.find(key);
		return it != cache.end() ? &it->second : nullptr;
	}
	auto&      front = GetTextureDescFront();
	const auto hash  = TextureDescKeyHash {}(key);
	auto&      slot  = front.slots[hash & (TextureDescFront::Slots - 1u)];
	if (slot.entry != nullptr && slot.generation == front.generation && slot.key == key) {
		return slot.entry;
	}
	const auto it = cache.find(key);
	if (it == cache.end()) {
		return nullptr;
	}
	slot = {key, &it->second, front.generation};
	return &it->second;
}

// KYTY_TEXTURE_MEMO=0, or KYTY_BISECT's texmemo:off: every binding is looked up again.
bool TextureMemoEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TEXTURE_MEMO");
		return value == nullptr || value[0] != '0';
	}();
	return enabled && !Bisect::TextureMemoOff();
}

// A plain sampled image: no storage, atomics, bindless heap, per-mip views or depth comparison.
bool TextureMemoEligible(const ShaderRecompiler::IR::ImageResource& resource) {
	return !resource.written && !resource.atomic && !resource.atomic64 && !resource.bindless &&
	       !resource.depth_compare &&
	       resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
}

// A lookup the memo may keep (TextureBinding::memo_generation): the image was there with the same
// backing, and no image over the descriptor's memory was made or dropped during the lookup (with
// texmemo:global, none anywhere). Results that came out of overlap resolution are left out: those
// can change with nothing registered or unregistered (an alias that ages out is freed by a later
// lookup), and the view base they adjust is the last candidate's.
// `range`: the descriptor's memory, whose stamp before the lookup was `stamp`.
void KeepForMemo(TextureBinding& binding, bool use_memo, bool same_backing, uint64_t generation,
                 uint64_t stamp, const GuestRange& range, const TextureCache& texture_cache) {
	if (!use_memo || !same_backing) {
		return;
	}
	if (Bisect::TextureMemoGlobal()) {
		if (texture_cache.ImageSetGeneration() == generation) {
			binding.memo_generation = generation;
			binding.memo_stamp      = texture_cache.RangeStamp(range.address, range.size);
		}
		return;
	}
	if (texture_cache.RangeStamp(range.address, range.size) == stamp) {
		// The generation as it is now, for the global test should the step change.
		binding.memo_generation = texture_cache.ImageSetGeneration();
		binding.memo_stamp      = stamp;
	}
}

// Whether a kept binding is still current for its descriptor's memory.
bool MemoCurrent(const TextureMemo& memo, const TextureCache::ImageDesc& desc,
                 const TextureCache& texture_cache) {
	if (memo.generation == 0) {
		return false;
	}
	if (Bisect::TextureMemoGlobal()) {
		return memo.generation == texture_cache.ImageSetGeneration();
	}
	return memo.stamp != 0 &&
	       memo.stamp == texture_cache.RangeStamp(desc.info.data.address, desc.info.data.size);
}

// Counted on the GPU thread, logged every 10 s ("Texture memo").
struct TextureMemoCounts {
	uint64_t lookups  = 0; // memo bindings asked for
	uint64_t found    = 0; // a memo held for the descriptor in this image set
	uint64_t reused   = 0; // used again as it was
	uint64_t refused  = 0; // the image had changed: looked up again
	uint64_t recorded = 0;
	// Why the others were looked up (round 78): a descriptor not seen before, one with nothing
	// kept, one kept before an image near its memory was made or dropped, or one whose image has
	// gone or must be bound again.
	uint64_t                              miss_new   = 0;
	uint64_t                              miss_none  = 0;
	uint64_t                              miss_stale = 0;
	uint64_t                              miss_image = 0;
	std::chrono::steady_clock::time_point last {};
};
TextureMemoCounts g_texture_memo_counts;

void ReportTextureMemo() {
	ReportImageChurnIfDue();
	auto&      c   = g_texture_memo_counts;
	const auto now = std::chrono::steady_clock::now();
	if (c.last.time_since_epoch().count() == 0) {
		c.last = now;
		return;
	}
	if (now - c.last < std::chrono::seconds(10)) {
		return;
	}
	LOGF("Texture memo (%.0f s): %" PRIu64 " sampled bindings, %" PRIu64
	     " found in the memo (%" PRIu64 " used again, %" PRIu64
	     " looked up again: the image had changed), %" PRIu64
	     " kept for reuse%s; the others: %" PRIu64 " new descriptors, %" PRIu64
	     " with nothing kept, %" PRIu64
	     " kept before an image near them was made or dropped%s, %" PRIu64
	     " whose image had gone\n",
	     std::chrono::duration<double>(now - c.last).count(), c.lookups, c.found, c.reused,
	     c.refused, c.recorded, TextureMemoEnabled() ? "" : " (off)", c.miss_new, c.miss_none,
	     c.miss_stale, Bisect::TextureMemoGlobal() ? " (anywhere: texmemo:global)" : "",
	     c.miss_image);
	c = {};
	c.last = now;
}

TextureDescKey MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
                                  const ShaderRecompiler::IR::DescriptorValue& value) {
	TextureDescKey key;
	std::copy_n(value.dwords.begin(), 8, key.dwords.begin());
	key.resource_class = static_cast<uint32_t>(resource.resource_class);
	key.numeric_class  = static_cast<uint32_t>(resource.numeric_class);
	key.dimension      = static_cast<uint32_t>(resource.dimension);
	key.mip_mode       = static_cast<uint32_t>(resource.mip_mode);
	key.mip_count      = resource.mip_count;
	key.conversion     = static_cast<uint32_t>(resource.conversion_format);
	key.swizzle        = resource.shader_swizzle;
	key.flags = (resource.read ? 1u : 0u) | (resource.written ? 2u : 0u) |
	            (resource.atomic ? 4u : 0u) | (resource.depth_compare ? 8u : 0u) |
	            (resource.cube ? 16u : 0u) | (resource.r128 ? 32u : 0u) |
	            (resource.bindless ? 64u : 0u) | (value.dword_count << 8u);
	return key;
}

bool TextureDescCacheEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TEXTURE_DESC_CACHE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

} // namespace

// Set while a bindless heap entry is resolved (ResolveBindlessKey). The guest rewrites heap
// entries as it streams textures, and one read mid-rewrite, or a freed one, decodes to a texture
// no draw binds: unaligned for its tiling (GT7, re-entering the garage). For such an entry the
// layout checks below leave the key on its placeholder instead of ending the emulator.
thread_local bool t_resolving_bindless = false;

static bool RejectBindlessTexture(const char* what, uint64_t address) {
	if (!t_resolving_bindless) {
		return false;
	}
	static std::atomic<uint32_t> reports {0};
	const auto                   report = reports.fetch_add(1, std::memory_order_relaxed);
	if (report < 16u || (report & (report + 1u)) == 0u) {
		LOGF("Bindless: heap entry for 0x%016" PRIx64 " left on its placeholder: %s (%" PRIu32
		     " so far)\n",
		     address, what, report + 1u);
	}
	return true;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              bool                                         memo) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	}

	auto&      cache     = TextureDescCache();
	const bool use_cache = TextureDescCacheEnabled();
	const auto key       = use_cache ? MakeTextureDescKey(resource, value) : TextureDescKey {};
	// The texture memo (TextureMemo): a draw's or a dispatch's plain sampled binding.
	const bool use_memo = memo && use_cache && TextureMemoEligible(resource) && TextureMemoEnabled();
	if (use_memo) {
		g_texture_memo_counts.lookups++;
	}
	if (use_cache) {
		if (auto* found = FindTextureDesc(key); found != nullptr) {
			auto& entry = *found;
			if (use_memo && MemoCurrent(entry.memo, entry.desc, texture_cache)) {
				// The image FindImage would find: RebindImages checks it is unchanged before it
				// uses the view again, and looks it up again otherwise.
				const auto* image = texture_cache.m_slot_images.try_get(entry.memo.id);
				if (image != nullptr && image->registered && !image->depth_id &&
				    !image->binding.needs_rebind) {
					g_texture_memo_counts.found++;
					TextureBinding binding {entry.memo.id, nullptr, entry.desc};
					binding.desc.view_info.base_level = entry.memo.base_level;
					binding.desc.view_info.base_layer = entry.memo.base_layer;
					binding.memo_view                 = entry.memo.view;
					return binding;
				}
				g_texture_memo_counts.miss_image++;
			} else if (use_memo) {
				(entry.memo.generation == 0 ? g_texture_memo_counts.miss_none
				                            : g_texture_memo_counts.miss_stale)++;
			}
			const auto generation   = texture_cache.ImageSetGeneration();
			const auto range        = entry.desc.info.data;
			const auto stamp        = texture_cache.RangeStamp(range.address, range.size);
			bool       same_backing = false;
			auto       binding      = FindResolvedTexture(
                resource, descriptor, entry.desc, entry.shader_conversion, entry.pixel_format,
                entry.view_format, entry.size, use_memo ? &same_backing : nullptr);
			KeepForMemo(binding, use_memo, same_backing, generation, stamp, range, texture_cache);
			return binding;
		}
	}
	if (use_memo) {
		g_texture_memo_counts.miss_new++;
	}

	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	const auto view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	// A T# no texture can be (the game's materials read descriptors it has already reused, as in
	// the cases below): sampled as a null texture, as the console returns something rather than
	// stopping, and logged the first times for each kind (round 81: two testers' races ended the
	// emulator on such T#s, one in the interior view).
	const auto null_texture = [&](std::atomic<uint32_t>& reports, const char* what) {
		const auto report = reports.fetch_add(1, std::memory_order_relaxed);
		if (report < 8u || (report & (report + 1u)) == 0u) {
			LOGF("Texture %s (base %u, last %u, max %u, %ux%ux%u, type %u, format %u, tile %u) at "
			     "0x%016" PRIx64 " bound as a null texture (%" PRIu32 " so far): "
			     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
			     what, base_level, last_level, max_mip, width, height,
			     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(type),
			     static_cast<uint32_t>(descriptor.Format()), static_cast<uint32_t>(tile), address,
			     report + 1u, descriptor.fields[0], descriptor.fields[1], descriptor.fields[2],
			     descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
			     descriptor.fields[6], descriptor.fields[7]);
		}
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return TextureBinding {id, nullptr, std::move(desc)};
	};
	// A first mip past the last: no view of any image. GT7 binds such a T# (its address and the
	// words after it are not a texture's) while it leaves a race, the car materials it still
	// draws reading descriptors already reused (2026-10-14, with tessellation drawing the cars).
	// Sampled as a null texture, as the console returns something rather than stopping.
	if (!multisampled && base_level > last_level) {
		if (RejectBindlessTexture("first mip past the last", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		const auto                   report = reports.fetch_add(1, std::memory_order_relaxed);
		if (report < 8u || (report & (report + 1u)) == 0u) {
			LOGF("Texture with its first mip past the last (base %u, last %u) at 0x%016" PRIx64
			     " bound as a null texture (%" PRIu32 " so far): "
			     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
			     base_level, last_level, address, report + 1u, descriptor.fields[0],
			     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
			     descriptor.fields[4], descriptor.fields[5], descriptor.fields[6],
			     descriptor.fields[7]);
		}
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	}
	// A multisample T# with mips or samples no multisample image has (an EXIT before round 81).
	if (multisampled &&
	    (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	     !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	     (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0)))) {
		if (RejectBindlessTexture("multisample view", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, "with a multisample view no image has");
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	// Round 94: a format no texture has, a component selector no texture has, or a minimum LOD
	// past the last mip (each an EXIT below): a heap entry of the bindless table read while the
	// guest rewrote it (format 160 ended a race with bindless textures on, 2026-10-19), or a T#
	// read after its reuse. Sampled as a null texture, as with a first mip past the last.
	if (!TextureFormatKnown(format)) {
		if (RejectBindlessTexture("format no texture has", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, "of a format no texture has");
	}
	if (!storage && !IsValidImageSwizzle(descriptor.DstSelXYZW())) {
		if (RejectBindlessTexture("component selector no texture has", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, "with a component selector no texture has");
	}
	if (descriptor.MinLod() > last_level * 256u) {
		if (RejectBindlessTexture("minimum LOD past the last mip", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, "with its minimum LOD past the last mip");
	}
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(format);
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		// The shader samples another class of numbers (float, uint, sint) than the format holds
		// (an EXIT before round 81).
		if (RejectBindlessTexture("numeric class of the format", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, resource.numeric_class == Prospero::TextureNumericClass::Float
		                                 ? "of a format a float sampler cannot read"
		                                 : "of a format an integer sampler cannot read");
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	// A tiling the format has no layout for, or more mips than the extent allows: no texture's
	// T#. A 20-kart race ended the emulator on k16_16UNorm 65x1 with 15 levels in swizzle mode 4
	// (2026-10-15). Sampled as a null texture, as with a first mip past the last.
	if (!multisampled && !TileTextureLayoutKnown(format, width, height,
	                                             volume ? depth : image_layers, physical_levels,
	                                             tile, volume)) {
		if (RejectBindlessTexture("no known layout", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		const auto                   report = reports.fetch_add(1, std::memory_order_relaxed);
		if (report < 8u || (report & (report + 1u)) == 0u) {
			LOGF("Texture with no known layout (format %u, %ux%ux%u, %u levels, tile %u) at "
			     "0x%016" PRIx64 " bound as a null texture (%" PRIu32 " so far): "
			     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
			     static_cast<uint32_t>(format), width, height, depth, physical_levels,
			     static_cast<uint32_t>(tile), address, report + 1u, descriptor.fields[0],
			     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
			     descriptor.fields[4], descriptor.fields[5], descriptor.fields[6],
			     descriptor.fields[7]);
		}
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	}
	if (levels > physical_levels) {
		const TileSurfaceDescription physical {
		    format, tile, volume ? TileSurfaceDimension::Dim3D : TileSurfaceDimension::Dim2D,
		    width, height, volume ? depth : 1u, physical_levels, image_layers};
		if (!ResolveTextureMipView(physical, !resource.r128 && descriptor.MetaCompress(),
		                           view_levels, levels, view_base)) {
			if (RejectBindlessTexture("mip view changes the physical layout", address)) {
				return {};
			}
			// Mips past the image's last (MAX_MIP) that no layout of its own mips can stand for:
			// as with a first mip past the last, the words are not a texture's. Testers' races
			// ended the emulator on such T#s (2026-10-18): 65x1x65 linear, mips 1 to 3 of an
			// image of one mip (the 65x1 of the kart race's garbage T# again), and 75x1x19, mip
			// 13 of one, in the interior view.
			static std::atomic<uint32_t> reports {0};
			return null_texture(reports, "with mips past its last");
		}
	}
	uint32_t      pitch = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			if (RejectBindlessTexture("multisample layout", address)) {
				return {};
			}
			static std::atomic<uint32_t> reports {0};
			return null_texture(reports, "with no known multisample layout");
		}
		size.size *= image_layers;
	} else {
		pitch = TileGetTexturePitch(format, width, tile);
		TileGetTextureTotalSize(format, width, height, volume ? depth : image_layers,
		                        physical_levels, tile, volume, size);
	}
	if (size.size == 0 || size.align == 0 ||
	    (address & (static_cast<uint64_t>(size.align) - 1u)) != 0) {
		// An empty layout, or an address its tiling cannot start at (an EXIT before round 81).
		if (RejectBindlessTexture("address not aligned for its tiling", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, "at an address its tiling cannot start at");
	}
	if (storage) {
		ValidateStorageTexture(resource, descriptor, size.size);
	}

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	// Raw sint stores and all image atomics run as uint operations on the texel bits
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && (resource.atomic ||
	                                               format == Prospero::BufferFormat::k32SInt)
	                                     ? vk::Format::eR32Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	// Round 156: memory past the guest's address space, for the texels or the metadata (a heap
	// entry read mid-rewrite: FindImage's range check ended a race on one at 0x3ec08000200).
	if (!desc.info.data.Valid() || !desc.info.metadata.range.ValidOrEmpty()) {
		if (RejectBindlessTexture("memory past the guest's address space", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, "in memory past the guest's address space");
	}
	// Round 94: a base layer past the image's layers (an EXIT in TextureViewInfo; "texture base
	// layer is out of bounds" ended a race with bindless textures on, a heap entry again).
	if (resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	    descriptor.BaseArray5() >= desc.info.resources.layers) {
		if (RejectBindlessTexture("base layer past its layers", address)) {
			return {};
		}
		static std::atomic<uint32_t> reports {0};
		return null_texture(reports, "with its base layer past its layers");
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;

	if (use_cache) {
		if (cache.size() >= 65536) {
			cache.clear();
			GetTextureDescFront().generation++;
		}
		cache.emplace(key, TextureDescEntry {desc, pixel_format, view_format, size.size,
		                                     shader_conversion});
	}
	const auto generation   = texture_cache.ImageSetGeneration();
	const auto range        = desc.info.data;
	const auto stamp        = texture_cache.RangeStamp(range.address, range.size);
	bool       same_backing = false;
	auto       binding = FindResolvedTexture(resource, descriptor, std::move(desc), shader_conversion,
	                                         pixel_format, view_format, size.size,
	                                         use_memo ? &same_backing : nullptr);
	KeepForMemo(binding, use_memo, same_backing, generation, stamp, range, texture_cache);
	return binding;
}

void RenderExecutor::ForgetTextureMemo(const ShaderRecompiler::IR::ImageResource&   resource,
                                       const ShaderRecompiler::IR::DescriptorValue& value) {
	if (auto* entry = FindTextureDesc(MakeTextureDescKey(resource, value)); entry != nullptr) {
		entry->memo = {};
	}
}

void RenderExecutor::RecordTextureMemo(const ShaderRecompiler::IR::ImageResource&   resource,
                                       const ShaderRecompiler::IR::DescriptorValue& value,
                                       const TextureBinding&                        binding) {
	if (binding.memo_generation == 0 || binding.image_view == nullptr || !TextureMemoEnabled() ||
	    binding.desc.type != TextureCache::BindingType::Texture || binding.desc.info.HasMetadata()) {
		return;
	}
	auto* entry = FindTextureDesc(MakeTextureDescKey(resource, value));
	if (entry == nullptr) {
		return;
	}
	if (!m_context.GetTextureCache().MayReuseTextureBinding(binding.image_id)) {
		entry->memo = {};
		return;
	}
	entry->memo = {binding.image_id,
	               binding.image_view,
	               binding.memo_generation,
	               binding.desc.view_info.base_level,
	               binding.desc.view_info.base_layer,
	               binding.memo_stamp};
	g_texture_memo_counts.recorded++;
}

// ResolveTexture's texture cache lookup for a description (FindImage may adjust it).
TextureBinding RenderExecutor::FindResolvedTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                                   const ShaderTextureResource&               descriptor,
                                                   TextureCache::ImageDesc desc,
                                                   bool shader_conversion, vk::Format pixel_format,
                                                   vk::Format view_format, uint32_t size,
                                                   bool* same_backing) {
	auto&      texture_cache       = m_context.GetTextureCache();
	const bool storage             = resource.written;
	auto       id                  = texture_cache.FindImage(desc, shader_conversion, same_backing);
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	// Round 124: the shader reads it as a single-sample image (an instruction of a single-sample
	// dimension over a multisample depth T#, ResourceMaterialization's DescriptorDimension).
	const bool single_sample_read =
	    resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray;
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
		// Round 124: the stencil of a multisample depth target read as single-sample (the
		// stencil plane's T# in replays): a copy of its first sample, as for the depth below.
		// Not for bindless heap entries: their slots take no image without guest memory (the
		// entry stays on its placeholder below).
		if (image->backing.samples > 1 && single_sample_read && !storage && !t_resolving_bindless) {
			id    = texture_cache.SingleSampleStencil(id);
			image = &texture_cache.GetImage(id);
		}
	} else if (image->info.IsDepth()) {
		// Round 94: a bindless heap entry the image it found cannot be viewed through (each an
		// EXIT below) is left on its placeholder, as with the layout checks in ResolveTexture.
		if ((storage || !(IsSupportedSampledDepthResource(resource) &&
		                  IsSupportedDepthTextureEncoding(descriptor, resource.r128) &&
		                  IsSupportedSampledDepthView(image->info.pixel_format, pixel_format,
		                                              descriptor.DstSelXYZW()))) &&
		    RejectBindlessTexture("a depth image it cannot view", descriptor.Base40())) {
			return {};
		}
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format, size);
		// Round 122: a multisample depth target read by an instruction of a single-sample
		// dimension (EQAA depth in replays: the light sprites' occlusion test, depth-based
		// effects) reads a copy of its first sample, as the console's texture unit does
		// (TextureCache::SingleSampleDepth). Round 124: the shader's dimension is that
		// single-sample one now (before, specialization gave it the T#'s multisample one, so
		// rounds 122 and 123 never made the copy).
		if (image->backing.samples > 1 && single_sample_read && !t_resolving_bindless) {
			id    = texture_cache.SingleSampleDepth(id);
			image = &texture_cache.GetImage(id);
		}
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		if (!IsSupportedSampledColorView(image->info.pixel_format, pixel_format,
		                                 descriptor.DstSelXYZW()) &&
		    RejectBindlessTexture("an image it cannot view", descriptor.Base40())) {
			return {};
		}
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
		// Round 124 (Franseven's case): a texture smaller than the image at its address, with the
		// same memory and tiling (text drawn into part of a wider target), reads a copy of the
		// image's corner: a view of the image would stretch its coordinates over the whole
		// image. Not for bindless heap entries (their slots keep a view past the draw).
		if (!t_resolving_bindless) {
			if (const auto copy_id = texture_cache.SubExtentCopy(id, desc.info); copy_id != id) {
				id    = copy_id;
				image = &texture_cache.GetImage(id);
			}
		}
	}
	if (image->backing.samples > 1 && single_sample_read) {
		// Round 124: a multisample image left for a single-sample binding (a colour image under a
		// depth-tiled multisample T#, or a depth image SingleSampleDepth or SingleSampleStencil
		// made no copy of). The shader's image is a single-sample one, and a multisample image
		// bound to it is a mismatch Vulkan leaves undefined (wrong reads or a lost device): a
		// null texture instead (a bindless entry stays on its placeholder).
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Multisample depth read as single-sample: 0x%012" PRIx64 " %ux%u fmt %d, %u "
			     "samples, %u layers, %u levels, %s%s: no single-sample copy, a null texture "
			     "instead\n",
			     image->info.data.address, image->info.extent.width, image->info.extent.height,
			     static_cast<int>(image->info.pixel_format), image->backing.samples,
			     image->info.resources.layers, image->info.resources.levels,
			     image->info.IsDepth() ? "a depth image" : "a colour image",
			     stencil_association ? " (its stencil)" : "");
		}
		if (RejectBindlessTexture("a multisample image read as single-sample",
		                          descriptor.Base40())) {
			return {};
		}
		if (same_backing != nullptr) {
			*same_backing = false;
		}
		auto null_desc     = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                       : TextureCache::BindingType::Texture);
		const auto null_id = texture_cache.FindImage(null_desc);
		return {null_id, nullptr, std::move(null_desc)};
	}
	return {id, nullptr, std::move(desc)};
}

// Round 142: KYTY_CUBE_EDGE_CLAMP=0 turns it off. A cube map is sampled here as a 2D array of
// its faces (face coordinates and face index from V_CUBE*), so the sampler's address mode applies
// inside each face. The console's cube addressing ignores it and filters across the face edges;
// with WRAP (what a sky's sampler often has, since the console ignores it) bilinear filtering at
// an edge blends in the face's opposite edge: a thin line along every face edge, in the sky and
// in the reflections. CLAMP_LAST_TEXEL keeps each face's filtering inside it.
static bool CubeEdgeClamp() {
	static const bool on = [] {
		const char* value   = std::getenv("KYTY_CUBE_EDGE_CLAMP");
		const bool  enabled = value == nullptr || value[0] != '0';
		LOGF("Cube maps: samplers used only with cube maps %s at the face edges "
		     "(KYTY_CUBE_EDGE_CLAMP)\n",
		     enabled ? "clamp" : "keep the guest's address mode");
		return enabled;
	}();
	return on;
}

// Whether every image this sampler is paired with is a cube map.
static bool SamplesCubesOnly(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                             uint32_t                                        sampler) {
	bool any = false;
	for (const auto& pair: program.info.sampled_pairs) {
		if (pair.sampler != sampler) {
			continue;
		}
		if (pair.image >= program.info.images.size() || !program.info.images[pair.image].cube) {
			return false;
		}
		any = true;
	}
	return any;
}

static vk::Sampler NativeSampler(RenderContext&                                  context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index, const ShaderRecompiler::IR::DescriptorValue& value,
                                 bool big_depth_image, bool normalize_unnormalized) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.bindless && CubeEdgeClamp() && SamplesCubesOnly(program, index)) {
		// CLAMP_X and CLAMP_Y (bits 0-2 and 3-5) to CLAMP_LAST_TEXEL (2).
		descriptor.fields[0] = (descriptor.fields[0] & ~0x3fu) | 2u | (2u << 3u);
	}
	if (normalize_unnormalized) {
		// The stage's copy divides its coordinates by the mip's size (SpirvNormalizedExplicitLod):
		// FORCE_UNNORMALIZED off, the guest's LOD range and clamps as they are.
		descriptor.fields[0] &= ~(1u << 15u);
	}
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	} else if (Bisect::PassesDepthCompare(big_depth_image)) {
		// KYTY_BISECT lit:big / lit:all: the comparison always passes (ALWAYS, 7).
		descriptor.fields[0] |= 0x7u << 12u;
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

// The flat slots the refresh left to the GPU (SrtGpuFill): their guest bytes are copied, where
// the buffer cache holds them, into the uploaded flat buffer before the command, behind the
// writes that produced them. Reading them on the host instead waited for the GPU (2.8 ms a frame
// for one dispatch in the jungle).
static void FillFlatSlotsOnGpu(RenderContext& context, const vk::DescriptorBufferInfo& flat,
                               std::span<const ShaderRecompiler::IR::SrtGpuFill> fills) {
	if (fills.empty()) {
		return;
	}
	auto& cache   = context.GetBufferCache();
	auto& command = context.GetCommandScheduler().Current();
	command.EndRendering();
	const auto      recorder = command.Recorder();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                         nullptr);
	for (const auto& fill: fills) {
		auto [buffer, offset] = cache.ObtainBuffer(fill.address, sizeof(uint32_t), false);
		(void)offset;
		const vk::BufferCopy copy {buffer->Offset(fill.address),
		                           flat.offset + uint64_t {fill.flat_offset} * sizeof(uint32_t),
		                           sizeof(uint32_t)};
		recorder.copyBuffer(buffer->Handle(), flat.buffer, 1, &copy);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eUniformRead;
	recorder.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                         vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                         nullptr);
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (!image.binding.is_target) {
		NoteBindlessStateChange(image);
	}
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			if (image->binding.is_target) {
				NoteBindlessStateChange(*image);
			}
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

// The T#s of keys [first, first + count) of a bindless heap whose keys' T#s are `stride` bytes
// apart from `table` on: one read when they are contiguous, one read of the covering span
// otherwise (a T# inside a larger per-key record).
static bool ReadHeapDescriptors(uint64_t table, uint32_t stride, uint64_t first, uint64_t count,
                                std::array<uint32_t, 8>* out) {
	if (count == 0) {
		return true;
	}
	if (stride == 32u) {
		return Libs::LibKernel::Memory::TryReadBacking(table + first * 32u, out, count * 32u);
	}
	thread_local std::vector<uint8_t> span;
	const uint64_t bytes = (count - 1u) * stride + 32u;
	span.resize(bytes);
	if (!Libs::LibKernel::Memory::TryReadBacking(table + first * stride, span.data(), bytes)) {
		return false;
	}
	for (uint64_t i = 0; i < count; i++) {
		std::memcpy(out[i].data(), span.data() + i * stride, 32u);
	}
	return true;
}

// Keys whose T# lies within `span` bytes of the table start.
static uint64_t HeapEntriesInSpan(uint64_t span, uint32_t stride) {
	return span < 32u ? 0 : (span - 32u) / stride + 1u;
}

// Diagnostics: the first time a draw uses a bindless heap,
// read its texture descriptors and log how many are valid, of which types, and roughly how
// much memory they would take as host textures. Nothing is created.
static void SurveyBindlessHeap(const ShaderRecompiler::IR::BindlessHeapUse& use) {
	constexpr uint64_t MaxEntries = 1u << 20u;
	const uint64_t     span       = use.size > use.table_offset ? use.size - use.table_offset : 0;
	const uint64_t     entries =
	    std::min<uint64_t>(HeapEntriesInSpan(span, use.table_stride), MaxEntries);
	uint64_t valid = 0, bytes = 0, unreadable = 0;
	std::array<uint64_t, 16> types {};
	std::vector<std::array<uint32_t, 8>> chunk(1024u);
	for (uint64_t first = 0; first < entries; first += chunk.size()) {
		const auto count = std::min<uint64_t>(chunk.size(), entries - first);
		if (!ReadHeapDescriptors(use.base + use.table_offset, use.table_stride, first, count,
		                         chunk.data())) {
			unreadable += count;
			continue;
		}
		for (uint64_t i = 0; i < count; i++) {
			ShaderRecompiler::IR::DescriptorValue value {};
			value.dword_count = 8u;
			std::copy_n(chunk[i].begin(), 8, value.dwords.begin());
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
			if (descriptor.IsNull()) {
				continue;
			}
			valid++;
			types[static_cast<uint32_t>(descriptor.Type()) & 15u]++;
			const uint64_t width  = static_cast<uint64_t>(descriptor.Width5()) + 1u;
			const uint64_t height = static_cast<uint64_t>(descriptor.Height5()) + 1u;
			const uint64_t bpe    = std::max<uint32_t>(Prospero::NumBytesPerElement(descriptor.Format()), 1u);
			bytes += width * height * bpe * 4u / 3u;
		}
	}
	std::string histogram;
	for (uint32_t t = 0; t < types.size(); t++) {
		if (types[t] != 0) {
			histogram += fmt::format(" t{}={}", t, types[t]);
		}
	}
	LOGF("Bindless heap: base=0x%016" PRIx64 " size=0x%" PRIx64 " offset=%u entries=%" PRIu64
	     " valid=%" PRIu64 " unreadable=%" PRIu64 " est=%" PRIu64 "MB types:%s\n",
	     use.base, use.size, use.table_offset, entries, valid, unreadable, bytes >> 20u,
	     histogram.c_str());
}

static uint32_t BindlessBindingFor(const ShaderRecompiler::IR::ImageResource& resource) {
	using ShaderRecompiler::Decoder::ImageDimension;
	if (resource.cube) {
		return ShaderRecompiler::IR::BindlessImagesCube;
	}
	if (resource.dimension == ImageDimension::Dim3D) {
		return ShaderRecompiler::IR::BindlessImages3D;
	}
	if (resource.dimension == ImageDimension::Dim2DArray ||
	    resource.dimension == ImageDimension::Dim2DMsaaArray) {
		return ShaderRecompiler::IR::BindlessImages2DArray;
	}
	return ShaderRecompiler::IR::BindlessImages2D;
}

// A heap entry the bindless array of this binding can hold: a float-sampled, single-sample
// texture of a matching shape. Anything else stays the placeholder rather than reaching
// ResolveTexture, which exits on descriptors a draw would never bind that way.
static bool BindlessCompatible(const ShaderTextureResource& descriptor, uint32_t binding,
                               bool runtime = false) {
	// Round 94: a format no texture has, which TextureGetSurfaceFormatInfo would end on.
	if (descriptor.IsNull() || !TextureFormatKnown(descriptor.Format())) {
		return false;
	}
	const auto numeric_class = Prospero::SampledTextureNumericClass(descriptor.Format());
	if (runtime) {
		// A runtime image's loads read integer textures through their own aliases; nothing
		// converts a format in the shader for it.
		if (numeric_class == Prospero::TextureNumericClass::Unsupported ||
		    TextureGetSurfaceFormatInfo(descriptor.Format()).conversion_format !=
		        Prospero::BufferFormat::kInvalid) {
			return false;
		}
	} else if (numeric_class != Prospero::TextureNumericClass::Float) {
		return false;
	}
	switch (descriptor.Type()) {
		case Prospero::ImageType::kColor2D:
			return binding == ShaderRecompiler::IR::BindlessImages2D ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray;
		case Prospero::ImageType::kColor2DArray:
			return binding == ShaderRecompiler::IR::BindlessImages2D ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray;
		case Prospero::ImageType::kCube: return binding == ShaderRecompiler::IR::BindlessImagesCube;
		case Prospero::ImageType::kColor3D: return binding == ShaderRecompiler::IR::BindlessImages3D;
		default: return false;
	}
}

bool RenderExecutor::ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	auto&                                 table = m_context.GetBindlessTable();
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count  = 8u;
	const auto address = heap.base + heap.table_offset + static_cast<uint64_t>(key) * heap.table_stride;
	heap.settled[key]  = 1;
	heap.stale[key]    = 0;
	heap.descriptors[key] = {};
	if (heap.runtime) {
		// The T# a shader requested (BindlessTable::RuntimeKey), not a guest heap entry.
		std::copy_n(heap.runtime_dwords[key].begin(), 8, value.dwords.begin());
	} else if (!Libs::LibKernel::Memory::TryReadBacking(address, value.dwords.data(), 32u)) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	std::copy_n(value.dwords.data(), 8, heap.descriptors[key].begin());
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	if (!BindlessCompatible(descriptor, heap.binding, heap.runtime)) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	auto resource = heap.resource;
	if (heap.runtime) {
		resource.numeric_class = Prospero::SampledTextureNumericClass(descriptor.Format());
	}
	t_resolving_bindless = true;
	auto binding         = ResolveTexture(resource, value);
	t_resolving_bindless = false;
	auto& texture_cache = m_context.GetTextureCache();
	auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty()) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	BindImage(binding.image_id, false);
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image != nullptr && !image->registered) {
		// Dropped by the cache (its destruction waits for the GPU): a slot holding its view would
		// outlive it, since its unregistration is over. The key is asked for again next frame.
		heap.settled[key] = 0;
		table.SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
		return false;
	}
	const auto slot = table.AllocateSlot(heap.binding);
	if (image == nullptr || view == nullptr || slot == 0) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
	table.WriteSlot(heap.binding, slot, view, layout);
	heap.slots[key]         = slot;
	heap.images[key]        = binding.image_id;
	image->bindless_pinned  = true;
	image->usage.texture    = true;
	heap.resolved.push_back(binding.image_id);
	heap.unchecked.push_back(binding.image_id);
	table.AddImageReference(binding.image_id, heap, key);
	table.SetTranslation(heap, key, slot);
	return true;
}

// Engine-method counters (SyncBindlessHeaps), logged and reset with the heap statistics.
struct BindlessEngineStats {
	std::atomic<uint64_t> syncs {0};
	std::atomic<uint64_t> written_bytes {0};
	std::atomic<uint64_t> view_changes {0};
	std::atomic<uint64_t> resource_changes {0};
	std::atomic<uint64_t> resettled {0};
	std::atomic<uint64_t> eager_settled {0};
	std::atomic<uint64_t> eager_failed {0};
	std::atomic<uint64_t> eager_paused {0};
	std::atomic<uint64_t> requested {0}; // keys draws asked for through feedback
	std::atomic<uint64_t> released {0};  // rewritten keys no draw had sampled: left to E3
	std::atomic<uint64_t> eager_capped {0}; // keys E3 stopped settling (MaxEagerSettles)
	std::atomic<uint64_t> kept_stale {0};   // view changes of unsampled keys left as they were
	std::atomic<uint64_t> stale_settled {0}; // such keys settled again once a draw sampled them
};
constexpr uint8_t MaxEagerSettles = 2;
static BindlessEngineStats g_bindless_engine_stats;

// With the engine method, every entry the guest has written is settled before draws ask for it
// (E3), at most KYTY_BINDLESS_ENGINE_BUDGET (default 256) per presented frame. On by default;
// KYTY_BINDLESS_ENGINE_EAGER=0 settles keys only when draws ask for them.
// Eager settling (E3) is off by default since 2026-10-08: in a new game it paused under the
// texture cache's bound for nearly the whole session, yet kept the ~3 GB it settled early (heap
// textures 9.3 GB against 6.0 GB without it, peak video memory 20.2 against 17.1 GB), with as
// many late requests (544 against 620 in the first 5,500 frames) and the same or better frame
// rate (prompt 21.6 against 22.1 fps). Textures load when a draw samples them; rewrites of keys
// already settled are still followed (E2). KYTY_BINDLESS_ENGINE_EAGER=1 turns it on.
static bool BindlessEngineEager() {
	static const bool eager = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_EAGER");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return eager;
}

// KYTY_BINDLESS_ENGINE_BUDGET_MB: the image memory eager settling may touch per presented frame
// (default 32). A count alone let a new area's load settle ~9,800 textures in 300 frames, large
// ones uploaded and detiled together, and the frames stalled; engines stream with a bandwidth.
static uint64_t BindlessEngineBudgetBytes() {
	static const uint64_t bytes = [] {
		const char* value  = std::getenv("KYTY_BINDLESS_ENGINE_BUDGET_MB");
		const auto  parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		return (parsed != 0 ? parsed : 32ull) << 20u;
	}();
	return bytes;
}

static uint32_t BindlessEngineBudget() {
	static const uint32_t budget = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_BUDGET");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
		return parsed != 0 ? static_cast<uint32_t>(parsed) : 256u;
	}();
	return budget;
}

// KYTY_BINDLESS_ENGINE_VERIFY=1 (with KYTY_BINDLESS_ENGINE=1): the rolling scan keeps running and
// logs every rewritten entry the write watch did not report.
static bool BindlessEngineVerify() {
	static const bool verify = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_VERIFY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return verify;
}

void RenderExecutor::ReleaseBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	auto&      table         = m_context.GetBindlessTable();
	auto&      texture_cache = m_context.GetTextureCache();
	const auto old_image     = heap.images[key];
	if (table.ReleaseKey(heap, key)) {
		if (auto* image = texture_cache.m_slot_images.try_get(old_image);
		    image != nullptr && image->bindless_pinned) {
			NoteBindlessStateChange(*image);
			image->bindless_pinned = false;
			if (BindlessEngineEnabled()) {
				texture_cache.ReleaseDroppedByGuest(old_image);
			}
		}
	}
}

bool RenderExecutor::ResettleBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	ReleaseBindlessKey(heap, key);
	return ResolveBindlessKey(heap, key);
}

void RenderExecutor::SyncBindlessHeaps() {
	auto& table = m_context.GetBindlessTable();
	if (!BindlessEngineEnabled() || !table.Enabled()) {
		return;
	}
	if (table.HasWrittenRanges()) {
		SyncWrittenBindlessEntries();
	}
	if (BindlessEngineEager()) {
		SettleBindlessHeaps();
	}
}

void RenderExecutor::SettleBindlessHeaps() {
	auto& table = m_context.GetBindlessTable();
	auto& stats = g_bindless_engine_stats;
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame != m_bindless_eager_frame) {
		m_bindless_eager_frame = frame;
		m_bindless_eager_left  = BindlessEngineBudget();
		m_bindless_eager_bytes = BindlessEngineBudgetBytes();
		// Settling ahead stops while the device is at 80 % of its budget, below the 90 % at which
		// the texture cache starts evicting, so the two do not chase each other. A new game's
		// jungle load filled the 32 GB card when only the images' own threshold was checked.
		m_bindless_eager_device_full = m_context.GetGraphics().DeviceMemoryAtLeast(80);
		// Once a frame: stale keys a draw has sampled since are settled again (finer mips).
		uint32_t stale_settled = 0;
		for (auto& heap: table.Heaps()) {
			if (heap.runtime) {
				continue;
			}
			std::erase_if(heap.stale_keys, [&](uint32_t key) {
				if (heap.stale[key] == 0) {
					return true; // settled again meanwhile
				}
				if (m_bindless_eager_left == 0 || !table.WasSampled(heap, key)) {
					return false;
				}
				m_bindless_eager_left--;
				heap.stale[key] = 0;
				stale_settled += ResettleBindlessKey(heap, key) ? 1u : 0u;
				return true;
			});
		}
		stats.stale_settled.fetch_add(stale_settled, std::memory_order_relaxed);
	}
	// Engine method, E3: entries no draw has asked for yet are settled too, as an engine creates a
	// texture's descriptor when it loads the texture. A pass stops at the budget and goes on from
	// its cursor at the next submission; the request path covers what is not settled yet. Paused
	// while the texture cache is short of memory, so eviction and settling do not chase each other.
	if (std::ranges::all_of(table.Heaps(), [](const auto& heap) { return heap.eager_complete; })) {
		return;
	}
	if (m_context.GetTextureCache().UnderPressure() || m_bindless_eager_device_full) {
		stats.eager_paused.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (m_bindless_eager_left == 0) {
		return;
	}
	KYTY_PROFILER_FUNCTION();
	constexpr uint32_t Window   = 2048;
	uint32_t&          budget   = m_bindless_eager_left;
	uint32_t           settled  = 0;
	uint32_t           failed   = 0;
	uint32_t           capped   = 0;
	for (auto& heap: table.Heaps()) {
		if (heap.eager_complete || budget == 0) {
			continue;
		}
		const auto keys = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		const auto base = heap.base + heap.table_offset;
		uint32_t   key  = heap.eager_cursor < keys ? heap.eager_cursor : 0u;
		uint32_t   seen = 0;
		bool       marked = false;
		while (seen < keys && budget != 0) {
			const auto count = std::min(Window, keys - key);
			m_bindless_window.resize(count);
			uint32_t   looked   = count;
			const bool readable =
			    ReadHeapDescriptors(base, heap.table_stride, key, count, m_bindless_window.data());
			for (uint32_t i = 0; readable && i < count; i++) {
				if (budget == 0) {
					looked = i;
					break;
				}
				if (heap.settled[key + i] != 0) {
					continue;
				}
				if (heap.eager_count[key + i] >= MaxEagerSettles) {
					continue; // its image keeps being replaced: the request path has it
				}
				ShaderTextureResource descriptor {};
				std::copy_n(m_bindless_window[i].data(), 8, descriptor.fields);
				if (descriptor.IsNull() || !BindlessCompatible(descriptor, heap.binding)) {
					continue;
				}
				budget--;
				if (++heap.eager_count[key + i] == MaxEagerSettles) {
					capped++;
				}
				if (ResolveBindlessKey(heap, key + i)) {
					if (const auto* image =
					        m_context.GetTextureCache().m_slot_images.try_get(heap.images[key + i]);
					    image != nullptr) {
						const auto size = image->AccountedSize();
						m_bindless_eager_bytes -= std::min(m_bindless_eager_bytes, size);
						if (m_bindless_eager_bytes == 0) {
							budget = 0; // this frame's bandwidth is spent
						}
					}
					// Not asked for by a draw: marked until one samples it (WasSampled), which also
					// lets the usage probe count it as unused under memory pressure.
					table.MarkUnsampled(heap, key + i);
					marked = true;
					settled++;
				} else {
					failed++;
				}
			}
			seen += looked;
			key = key + looked >= keys ? 0u : key + looked;
		}
		heap.eager_cursor = key;
		heap.eager_complete = seen >= keys && budget != 0;
		if (marked) {
			table.FlushFeedback(heap);
		}
	}
	stats.eager_settled.fetch_add(settled, std::memory_order_relaxed);
	stats.eager_failed.fetch_add(failed, std::memory_order_relaxed);
	stats.eager_capped.fetch_add(capped, std::memory_order_relaxed);
}

void RenderExecutor::SyncWrittenBindlessEntries() {
	auto& table = m_context.GetBindlessTable();
	KYTY_PROFILER_FUNCTION();
	// Engine method, E2: the pages the guest wrote since the last submission, already watched
	// again. Every entry in them is compared with the T# its key was settled from.
	table.TakeWrittenRanges(m_bindless_written);
	// T# bits that select a view of the same texture (RDNA 2 ISA 8.2.6); see LogBindlessHeapStats.
	constexpr std::array<uint32_t, 8> ViewMask {0u,          0x000fff00u, 0u,          0x0e0fffffu,
	                                            0x1fff0000u, 0x027fff00u, 0x000003ffu, 0u};
	uint32_t view_changes = 0, resource_changes = 0, resettled = 0, released = 0, kept = 0;
	for (auto& heap: table.Heaps()) {
		if (heap.runtime) {
			continue;
		}
		const auto heap_begin = heap.base + heap.table_offset;
		const auto keys       = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		const auto heap_end   = heap_begin + BindlessTable::HeapSpan(heap, keys);
		for (const auto& [begin, end]: m_bindless_written) {
			const auto from = std::max(begin, heap_begin);
			const auto to   = std::min(end, heap_end);
			if (from >= to) {
				continue;
			}
			heap.eager_complete = false; // entries may have become valid
			const auto stride = heap.table_stride;
			const auto first  = static_cast<uint32_t>((from - heap_begin) / stride);
			const auto last   = std::min(keys, static_cast<uint32_t>((to - heap_begin + stride - 1u) / stride));
			if (last <= first) {
				continue;
			}
			m_bindless_window.resize(last - first);
			if (!ReadHeapDescriptors(heap_begin, stride, first, last - first,
			                         m_bindless_window.data())) {
				continue;
			}
			for (uint32_t key = first; key < last; key++) {
				const auto& words = m_bindless_window[key - first];
				if (heap.descriptors[key] != words) {
					heap.eager_count[key] = 0; // the guest wrote it: E3 may settle it again
				}
				if (heap.settled[key] == 0 || heap.descriptors[key] == words) {
					continue;
				}
				bool resource = false;
				for (uint32_t i = 0; i < 8; i++) {
					resource |= ((heap.descriptors[key][i] ^ words[i]) & ~ViewMask[i]) != 0;
				}
				(resource ? resource_changes : view_changes)++;
				if (BindlessEngineEager() && !table.WasSampled(heap, key)) {
					// No draw has sampled it since it was settled.
					if (!resource) {
						// Streaming: the same texture with finer mips. The old view stays valid;
						// it is settled again when a draw samples the key.
						if (heap.stale[key] == 0) {
							heap.stale[key] = 1;
							heap.stale_keys.push_back(key);
						}
						kept++;
						continue;
					}
					// Another texture: the settle pass takes it within its budget, instead of
					// every rewrite of a load at once.
					ReleaseBindlessKey(heap, key);
					released++;
					continue;
				}
				resettled += ResettleBindlessKey(heap, key) ? 1u : 0u;
			}
		}
	}
	uint64_t bytes = 0;
	for (const auto& [begin, end]: m_bindless_written) bytes += end - begin;
	auto& stats = g_bindless_engine_stats;
	stats.syncs.fetch_add(1, std::memory_order_relaxed);
	stats.written_bytes.fetch_add(bytes, std::memory_order_relaxed);
	stats.view_changes.fetch_add(view_changes, std::memory_order_relaxed);
	stats.resource_changes.fetch_add(resource_changes, std::memory_order_relaxed);
	stats.resettled.fetch_add(resettled, std::memory_order_relaxed);
	stats.released.fetch_add(released, std::memory_order_relaxed);
	stats.kept_stale.fetch_add(kept, std::memory_order_relaxed);
}

uint32_t RenderExecutor::RevalidateBindlessKeys(uint32_t budget) {
	KYTY_PROFILER_FUNCTION();
	if (BindlessEngineEnabled() && !BindlessEngineVerify()) {
		return 0; // the write watch reports rewritten entries (SyncBindlessHeaps)
	}
	// The guest rewrites heap entries as it streams textures out and others in. A key settled
	// from the old T# kept sampling the old texture, and kept it pinned in the texture cache for
	// the rest of the run (in the jungle, 14-21 keys and 44-65 MB at any time). Each frame a
	// window of every heap is read back in one go and compared; a changed key is settled again
	// into a new slot, and an image no key refers to any more is unpinned, so the cache can
	// collect it. A heap of 64 Ki keys is covered in 32 frames.
	constexpr uint32_t Window = 2048;
	auto&              table         = m_context.GetBindlessTable();
	uint32_t           resolved      = 0;
	for (auto& heap: table.Heaps()) {
		const auto keys = static_cast<uint32_t>(heap.settled.size());
		if (keys == 0 || heap.runtime) {
			continue;
		}
		const uint32_t begin = heap.revalidate_cursor < keys ? heap.revalidate_cursor : 0u;
		const uint32_t count = std::min(Window, keys - begin);
		m_bindless_window.resize(count);
		const bool readable = ReadHeapDescriptors(heap.base + heap.table_offset, heap.table_stride,
		                                          begin, count, m_bindless_window.data());
		uint32_t next = begin + count;
		for (uint32_t i = 0; readable && i < count; i++) {
			const auto key = begin + i;
			if (heap.settled[key] == 0 || heap.descriptors[key] == m_bindless_window[i]) {
				continue;
			}
			if (resolved >= budget) {
				next = key; // look at it again next frame
				break;
			}
			if (BindlessEngineEnabled()) {
				static std::atomic<uint32_t> missed = 0;
				if (missed.fetch_add(1) < 64) {
					LOGF("Bindless engine: write not reported, heap=0x%" PRIx64 "+0x%x binding=%u "
					     "key=%u\n",
					     heap.base, heap.table_offset, heap.binding, key);
				}
			}
			resolved += ResettleBindlessKey(heap, key) ? 1u : 0u;
		}
		heap.revalidate_cursor = next >= keys ? 0u : next;
	}
	return resolved;
}

// Bytes of a T#'s mip chain from first_level down, ignoring tiling padding: an estimate.
static uint64_t MipChainBytes(const ShaderTextureResource& descriptor, uint32_t first_level) {
	const auto     format = descriptor.Format();
	const auto     block  = Prospero::BlockCompressedBytesPerBlock(format);
	const auto     texel  = block != 0 ? 0u : Prospero::NumBytesPerElement(format);
	const uint64_t width  = descriptor.Width5() + 1u;
	const uint64_t height = descriptor.Height5() + 1u;
	uint64_t       layers = descriptor.Type() == Prospero::ImageType::kColor2D
	                            ? 1u
	                            : static_cast<uint64_t>(descriptor.Depth()) + 1u;
	if (descriptor.Type() == Prospero::ImageType::kCube) layers = 6u;
	uint64_t bytes = 0;
	for (uint32_t level = first_level; level <= descriptor.MaxMip(); level++) {
		const auto w = std::max<uint64_t>(width >> level, 1u);
		const auto h = std::max<uint64_t>(height >> level, 1u);
		bytes += layers * (block != 0 ? ((w + 3u) / 4u) * ((h + 3u) / 4u) * block : w * h * texel);
	}
	return bytes;
}

// Diagnostics for the engine method (engine-method.md, E0): KYTY_BINDLESS_HEAP_STATS=1 logs every
// 300 presented frames, per heap:
// - valid: entries holding a texture this heap's view type can sample;
// - textures: distinct texture addresses among them (format aliases of one texture count once);
// - all_mb: their full mip chains; resident_mb: only the mips at or above each T#'s min LOD (what
//   the game has streamed in); sampled_mb: resident_mb of the textures draws asked for so far,
//   what translation on demand holds;
// - rewrites since the last line: view fields only (min LOD: streaming) or resource fields;
// - mip_stats: entries enabling the PS5's mip statistics;
// then the write watch's totals since the last line.
static void LogBindlessHeapStats(BindlessTable& table, uint64_t frame) {
	static const bool enabled = std::getenv("KYTY_BINDLESS_HEAP_STATS") != nullptr;
	static uint64_t   next    = 0;
	if (!enabled || frame < next) {
		return;
	}
	next = frame + 300;
	// T# bits that select a view of the same texture (RDNA 2 ISA 8.2.6): min LOD; dst_sel, base
	// and last level, BC swizzle; base array; min LOD warning, perf mod, mip-stats enable;
	// mip-stats slot and cache policy. Everything else describes the texture itself.
	constexpr std::array<uint32_t, 8> ViewMask {0u,          0x000fff00u, 0u,          0x0e0fffffu,
	                                            0x1fff0000u, 0x027fff00u, 0x000003ffu, 0u};
	static std::unordered_map<const BindlessTable::Heap*, std::vector<std::array<uint32_t, 8>>>
	                                     previous;
	std::vector<std::array<uint32_t, 8>> words;
	for (const auto& heap: table.Heaps()) {
		const auto entries =
		    std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		if (entries == 0) {
			continue;
		}
		words.resize(entries);
		if (!ReadHeapDescriptors(heap.base + heap.table_offset, heap.table_stride, 0, entries,
		                         words.data())) {
			continue;
		}
		uint32_t valid = 0, mip_stats = 0, settled = 0, view_changes = 0, resource_changes = 0;
		// Per texture address: the largest full chain and resident part seen, and whether a
		// sampled key refers to it.
		struct Texture {
			uint64_t all      = 0;
			uint64_t resident = 0;
			bool     sampled  = false;
		};
		std::unordered_map<uint64_t, Texture> textures;
		auto& last = previous[&heap];
		for (uint32_t key = 0; key < entries; key++) {
			settled += heap.settled[key] != 0 ? 1u : 0u;
			if (key < last.size() && last[key] != words[key]) {
				bool resource = false;
				for (uint32_t i = 0; i < 8; i++) {
					resource |= ((last[key][i] ^ words[key][i]) & ~ViewMask[i]) != 0;
				}
				(resource ? resource_changes : view_changes)++;
			}
			ShaderTextureResource descriptor {};
			std::copy_n(words[key].data(), 8, descriptor.fields);
			if (descriptor.IsNull() || !BindlessCompatible(descriptor, heap.binding)) {
				continue;
			}
			valid++;
			mip_stats += descriptor.MipStatsCntEn() ? 1u : 0u;
			const auto first =
			    std::max<uint32_t>(descriptor.BaseLevel(), descriptor.MinLod() >> 8u);
			auto& texture    = textures[descriptor.Base40()];
			texture.all      = std::max(texture.all, MipChainBytes(descriptor, 0));
			texture.resident = std::max(texture.resident, MipChainBytes(descriptor, first));
			texture.sampled |= heap.settled[key] != 0;
		}
		uint64_t all = 0, resident = 0, sampled = 0, sampled_textures = 0;
		for (const auto& [address, texture]: textures) {
			all += texture.all;
			resident += texture.resident;
			if (texture.sampled) {
				sampled += texture.resident;
				sampled_textures++;
			}
		}
		last = words;
		LOGF("Bindless heap stats: frame=%" PRIu64 " base=0x%" PRIx64 "+0x%x binding=%u entries=%u "
		     "valid=%u textures=%zu sampled_textures=%" PRIu64 " all_mb=%" PRIu64
		     " resident_mb=%" PRIu64 " sampled_mb=%" PRIu64 " settled=%u rewrites_view=%u "
		     "rewrites_resource=%u mip_stats=%u\n",
		     frame, heap.base, heap.table_offset, heap.binding, entries, valid, textures.size(),
		     sampled_textures, all >> 20u, resident >> 20u, sampled >> 20u, settled, view_changes,
		     resource_changes, mip_stats);
	}
	auto& stats = g_bindless_engine_stats;
	LOGF("Bindless engine stats: frame=%" PRIu64 " syncs=%" PRIu64 " written=%" PRIu64
	     " KiB view=%" PRIu64 " resource=%" PRIu64 " resettled=%" PRIu64 " eager_settled=%" PRIu64
	     " eager_failed=%" PRIu64 " eager_paused=%" PRIu64 " requested=%" PRIu64
	     " released=%" PRIu64 " eager_capped=%" PRIu64 " unregistered_keys=%" PRIu64
	     " kept_stale=%" PRIu64 " stale_settled=%" PRIu64 "\n",
	     frame, stats.syncs.exchange(0), stats.written_bytes.exchange(0) >> 10u,
	     stats.view_changes.exchange(0), stats.resource_changes.exchange(0),
	     stats.resettled.exchange(0), stats.eager_settled.exchange(0),
	     stats.eager_failed.exchange(0), stats.eager_paused.exchange(0),
	     stats.requested.exchange(0), stats.released.exchange(0), stats.eager_capped.exchange(0),
	     table.TakeUnregisteredKeys(), stats.kept_stale.exchange(0),
	     stats.stale_settled.exchange(0));
}

void RenderExecutor::ResolveBindlessRequests() {
	KYTY_PROFILER_FUNCTION();
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame == m_bindless_frame) {
		return;
	}
	auto& table     = m_context.GetBindlessTable();
	auto& scheduler = m_context.GetCommandScheduler();
	// The flags come from a snapshot recorded in an earlier frame. Until the GPU has executed
	// it, check again at the next bindless draw instead of waiting.
	if (table.SnapshotRecorded() && !table.SnapshotReady(scheduler)) {
		return;
	}
	m_bindless_frame = frame;
	if (!table.SnapshotRecorded()) {
		table.RecordFeedbackSnapshot(scheduler);
		m_bindless_snapshot_frame = frame;
		return;
	}
	table.ConsumeSnapshot();
	const auto snapshot_frame = m_bindless_snapshot_frame;
	// Each texture may upload and detile; spread first sight of a scene over a few frames.
	constexpr uint32_t Budget   = 128;
	uint32_t           resolved = RevalidateBindlessKeys(Budget);
	uint32_t           requested = 0;
	static std::atomic<uint32_t> frames_logged = 0;
	const bool log_frame = frames_logged.fetch_add(1) < 12;
	if (const auto word0 = m_context.GetBindlessTable().TakeWordZero(); word0 != 0) {
		static std::atomic<uint32_t> keys_logged = 0;
		if (keys_logged.fetch_add(1) < 64) {
			LOGF("Bindless out-of-range key: frame=%" PRIu64 " key=%u (0x%08x)\n", frame,
			     word0 & 0x7fffffffu, word0);
		}
	}
	// T#s shaders loaded at run time and found missing from the runtime image table: each takes
	// a key of its binding's runtime heap and is resolved like a heap key. A key not resolved now
	// stays pending in the table, and the next draw that reads it flags it.
	m_runtime_image_requests.clear();
	table.TakeRuntimeRequests(m_runtime_image_requests);
	uint32_t runtime_requested = 0, runtime_resolved = 0;
	for (const auto& [dwords, binding]: m_runtime_image_requests) {
		auto* heap = table.RuntimeHeap(binding);
		if (heap == nullptr) {
			continue;
		}
		const auto key = table.RuntimeKey(*heap, dwords);
		if (key == UINT32_MAX || heap->settled[key] != 0) {
			continue;
		}
		runtime_requested++;
		if (resolved >= Budget) {
			continue;
		}
		const bool ok = ResolveBindlessKey(*heap, key);
		resolved += ok ? 1u : 0u;
		runtime_resolved += ok ? 1u : 0u;
	}
	if (runtime_requested != 0) {
		static std::atomic<uint32_t> runtime_logged = 0;
		const auto                   logged_count   = runtime_logged.fetch_add(1);
		if (logged_count < 32 || (logged_count & (logged_count + 1u)) == 0u) {
			uint32_t keys = 0;
			for (uint32_t b = 0; b < BindlessTable::ImageArrays; b++) {
				if (const auto* heap = table.RuntimeHeap(b); heap != nullptr) {
					keys += heap->runtime_next_key;
				}
			}
			const auto line = fmt::format(
			    "Bindless runtime images: frame={} requested={} resolved={} ({} T#s so far)\n",
			    frame, runtime_requested, runtime_resolved, keys);
			std::fputs(line.c_str(), stdout);
			LOGF("%s", line.c_str());
		}
	}
	for (auto& heap: m_context.GetBindlessTable().Heaps()) {
		m_bindless_requests.clear();
		m_context.GetBindlessTable().TakeRequests(heap, m_bindless_requests);
		requested += static_cast<uint32_t>(m_bindless_requests.size());
		if (log_frame) {
			LOGF("Bindless feedback: frame=%" PRIu64 " binding=%u region=%u flagged=%zu\n",
			     frame, heap.binding, heap.region, m_bindless_requests.size());
		}
		for (const auto key: m_bindless_requests) {
			if (heap.settled[key] != 0) {
				continue;
			}
			if (resolved >= Budget) {
				break; // still pending: the next frame flags it again
			}
			resolved += ResolveBindlessKey(heap, key) ? 1u : 0u;
		}
	}
	Profiler::Add(Profiler::Counter::BindlessResolved, resolved);
	g_bindless_engine_stats.requested.fetch_add(requested, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged = 0;
	if (requested != 0 && logged.fetch_add(1) < 64) {
		LOGF("Bindless requests: frame=%" PRIu64 " requested=%u resolved=%u\n", frame,
		     requested, resolved);
	}
	UpdateBindlessUsageProbe(frame, snapshot_frame);
	LogBindlessHeapStats(table, frame);
	table.RecordFeedbackSnapshot(scheduler);
	m_bindless_snapshot_frame = frame;
}

// Bindless textures stay pinned in the texture cache while a heap key refers to them, and the
// guest's heaps refer to every texture it has streamed in: on 4-8 GB cards they grew to 2-5 GB
// and never shrank, until allocations failed (ISSUES #15). Under memory pressure a probe finds
// the ones no draw samples any more: it marks every resident key's feedback word, draws that
// sample a key overwrite its mark, and a snapshot taken ProbeFrames later lists the images whose
// keys all kept it. The texture cache may then free those like unpinned images; their keys go
// back to pending, and a draw that samples one again asks for it anew. KYTY_BINDLESS_EVICT=0
// disables the probe.
void RenderExecutor::UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_EVICT");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled) {
		return;
	}
	constexpr uint64_t ProbeFrames   = 16; // frames a texture must go unsampled
	constexpr uint64_t ProbeInterval = 32; // frames between the end of one probe and the next
	auto&              table         = m_context.GetBindlessTable();
	auto&              texture_cache = m_context.GetTextureCache();
	if (m_bindless_probe_frame != 0) {
		// The snapshot read here must have been recorded ProbeFrames after the marks.
		if (snapshot_frame < m_bindless_probe_frame + ProbeFrames) {
			return;
		}
		m_bindless_unused.clear();
		table.CollectUnusedImages(m_bindless_unused);
		texture_cache.SetBindlessEvictable(m_bindless_unused);
		m_bindless_probe_frame = 0;
		m_bindless_next_probe  = frame + ProbeInterval;
		static std::atomic<uint32_t> logged = 0;
		if (logged.fetch_add(1) < 32) {
			uint64_t bytes = 0;
			for (const auto id: m_bindless_unused) {
				if (const auto* image = texture_cache.m_slot_images.try_get(id); image != nullptr) {
					bytes += image->AccountedSize();
				}
			}
			LOGF("Bindless probe: frame=%" PRIu64 " unused=%zu (%" PRIu64 " MB) may be evicted\n",
			     frame, m_bindless_unused.size(), bytes >> 20u);
		}
		return;
	}
	// The heap's textures follow the guest (settled when it writes an entry, freed when it drops
	// them), so the probe evicts them only when the device itself is short of memory (4-8 GB
	// cards), not at the texture cache's bound: under a 6 GiB bound it evicted 27,915 of them in
	// 14 minutes, each settled again when drawn.
	if (frame >= m_bindless_next_probe && m_context.GetGraphics().DeviceMemoryAtLeast(90)) {
		table.ArmUsageProbe();
		m_bindless_probe_frame = frame;
	}
}

// Bindless samplers: the S# records of each sampler heap a draw indexes are mirrored into the
// sampler array of set 1, once per frame per heap, and the shader gets the region base and record
// count. Keys outside it, and heaps that cannot be read, use the default sampler in slot 0.
void RenderExecutor::PrepareBindlessSamplers(const ShaderStageRuntime& runtime,
                                             PreparedBindings&         prepared) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	if (snapshot.bindless_sampler_heaps.empty()) {
		return;
	}
	auto& table = m_context.GetBindlessTable();
	auto& cache = m_context.GetSamplerCache();
	{
		ShaderSamplerResource default_sampler;
		std::ranges::copy(ShaderRecompiler::IR::BindlessDefaultSampler, default_sampler.fields);
		table.WriteDefaultSampler(cache.GetSampler(default_sampler, false));
	}
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	constexpr uint64_t MaxRecords = 1024;
	for (const auto& use: snapshot.bindless_sampler_heaps) {
		uint32_t region  = 0;
		uint32_t entries = 0;
		if (table.SamplersEnabled() && use.sampler < program.info.samplers.size()) {
			const auto&    sampler = program.info.samplers[use.sampler];
			const uint32_t flags =
			    (sampler.depth_compare ? BindlessTable::SamplerDepthCompare : 0u) |
			    (sampler.force_point_filtering ? BindlessTable::SamplerPointFiltering : 0u) |
			    (sampler.integer_border ? BindlessTable::SamplerIntegerBorder : 0u);
			auto* heap = table.FindOrCreateSamplerHeap(use.base, use.table_offset, flags);
			if (heap->checked_frame != frame) {
				heap->checked_frame = frame;
				const uint64_t available =
				    use.size > use.table_offset ? (use.size - use.table_offset) / 16u : 0u;
				std::vector<std::array<uint32_t, 4>> records(std::min(available, MaxRecords));
				const uint64_t address = use.base + use.table_offset;
				const uint64_t bytes   = records.size() * sizeof(records[0]);
				if (!records.empty() &&
				    !m_context.GetBufferCache().HasGpuDirtyBytes(address, bytes) &&
				    Libs::LibKernel::Memory::TryReadBacking(address, records.data(), bytes)) {
					(void)table.MirrorSamplerHeap(*heap, records, cache);
				}
			}
			region  = heap->region;
			entries = region != 0 ? static_cast<uint32_t>(heap->records.size()) : 0u;
		}
		prepared.bindless_patches.push_back({use.mapping_offset, region, entries});
	}
}

void RenderExecutor::PrepareBindlessHeaps(const ShaderStageRuntime& runtime,
                                          PreparedBindings& prepared) {
	prepared.bindless_patches.clear();
	prepared.bindless_heaps.clear();
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	auto&       table    = m_context.GetBindlessTable();
	const bool  runtime_images =
	    std::ranges::any_of(program.info.images, &ShaderRecompiler::IR::ImageResource::runtime);
	if ((snapshot.bindless_heaps.empty() && snapshot.bindless_sampler_heaps.empty() &&
	     !runtime_images) ||
	    !table.Enabled()) {
		return;
	}
	PrepareBindlessSamplers(runtime, prepared);
	if (snapshot.bindless_heaps.empty() && !runtime_images) {
		return;
	}
	ResolveBindlessRequests();
	if (runtime_images) {
		// Every runtime heap of a binding the shader reads: its resolved images must be readable.
		for (const auto& image: program.info.images) {
			if (!image.runtime) {
				continue;
			}
			auto* heap = table.RuntimeHeap(BindlessBindingFor(image));
			if (heap != nullptr &&
			    std::ranges::find(prepared.bindless_heaps, heap) == prepared.bindless_heaps.end()) {
				prepared.bindless_heaps.push_back(heap);
			}
		}
	}
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.image >= program.info.images.size()) {
			continue;
		}
		const auto& resource = program.info.images[use.image];
		const uint64_t span  = use.size > use.table_offset ? use.size - use.table_offset : 0;
		const auto entries   = static_cast<uint32_t>(
		    std::min<uint64_t>(HeapEntriesInSpan(span, use.table_stride), 1u << 18u));
		auto* heap = table.FindOrCreateHeap(use.base, use.table_offset, use.table_stride,
		                                    BindlessBindingFor(resource), entries, resource);
		if (heap == nullptr) {
			continue; // region 0, count 0: every key samples the placeholder
		}
		if (BindlessEngineEnabled() && table.NeedsWatch(*heap)) {
			table.WatchHeap(*heap);
		}
		prepared.bindless_patches.push_back(
		    {use.mapping_offset, heap->region, std::min(entries, heap->entries)});
		prepared.bindless_heaps.push_back(heap);
	}
}

static const char* FrameGraphStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "cs";
		case ShaderType::Vertex: return "vs";
		case ShaderType::Local: return "ls";
		case ShaderType::TessellationControl: return "hs";
		case ShaderType::TessellationEvaluation: return "tes";
		case ShaderType::Mesh: return "ms";
		case ShaderType::Pixel: return "ps";
		default: return "?";
	}
}

// Diagnostics (the frame graph): a stage's textures (with their cache states), buffers and user
// data; the buffers and the user-data pointers are noted for the end-of-frame contents.
static void LogFrameGraphStage(RenderContext& context, const ShaderStageRuntime& runtime,
                               const PreparedBindings& prepared) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	const char* stage    = FrameGraphStage(program.stage);
	auto&       textures = context.GetTextureCache();
	auto&       buffers  = context.GetBufferCache();
	// Round 90: whether it reads memory through addresses (what the walk for such shaders keeps up
	// to date, RenderContext::PrepareBda, and the buffer lines below do not show).
	FrameGraph::Log(
	    "  %s %016" PRIx64 ": %zu textures, %zu buffers, %zu user data, %zu bindless heaps%s",
	    stage, program.shader_hash, program.info.images.size(), program.info.buffers.size(),
	    snapshot.user_data.size(), snapshot.bindless_heaps.size(),
	    program.info.uses_dma ? ", reads memory through addresses" : "");
	for (size_t i = 0; i < prepared.images.size() && i < program.info.images.size(); i++) {
		const auto& binding  = prepared.images[i];
		const auto& resource = program.info.images[i];
		const auto& info     = binding.desc.info;
		const auto& view     = binding.desc.view_info;
		const auto  state    = binding.image_id ? textures.DiagnosticImageState(binding.image_id)
		                                        : TextureCache::DiagnosticState {};
		const char  flags[]  = {state.gpu_written ? 'G' : '-', state.buffer_write ? 'B' : '-',
                              state.cpu_dirty ? 'C' : '-', '\0'};
		FrameGraph::Log(
		    "  %s tex[%zu] %s%s%s%s 0x%012" PRIx64 "+0x%" PRIx64
		    " %ux%ux%u layers %u mips %u view %u+%u/%u+%u fmt %d guest %u tile %u type %u [%s] "
		    "overlaps %u (%u GPU-written, 0x%012" PRIx64 ")",
		    stage, i, resource.written ? "write" : "read", resource.bindless ? " bindless" : "",
		    resource.runtime ? " runtime" : "",
		    binding.desc.type == TextureCache::BindingType::Storage ? " storage" : "",
		    info.data.address, info.data.size, info.extent.width, info.extent.height,
		    info.extent.depth, info.resources.layers, info.resources.levels, view.base_level,
		    view.level_count, view.base_layer, view.layer_count,
		    static_cast<int>(info.pixel_format), static_cast<uint32_t>(info.guest_format),
		    static_cast<uint32_t>(info.tile_mode), static_cast<uint32_t>(info.type),
		    state.valid ? flags : "none", state.overlaps, state.gpu_overlaps, state.first_overlap);
		// Round 123: the image the binding found, when its shape is not the T#'s (a glyph
		// texture viewed through an older image of another width at its address, a copy of a
		// depth image's first sample).
		if (state.valid &&
		    (state.data != info.data || state.extent != info.extent ||
		     state.host_width != std::max(info.extent.width, 1u) ||
		     state.host_height != std::max(info.extent.height, 1u) ||
		     state.levels < info.resources.levels || state.layers < info.resources.layers)) {
			FrameGraph::Log("  %s tex[%zu] found image 0x%012" PRIx64 "+0x%" PRIx64
			                " %ux%ux%u layers %u mips %u fmt %d, host %ux%u",
			                stage, i, state.data.address, state.data.size, state.extent.width,
			                state.extent.height, state.extent.depth, state.layers, state.levels,
			                static_cast<int>(state.format), state.host_width, state.host_height);
		}
		// A light grid the CPU wrote: does the GPU's copy hold what the guest wrote? (Shown as
		// the submission lands.)
		if (!resource.written && binding.image_id && info.pixel_format == vk::Format::eR32G32Uint &&
		    info.resources.layers > 1 && info.data.size <= 0x40000u &&
		    FrameGraph::ClaimGridCheck(info.data.address)) {
			textures.CompareWithGuest(binding.image_id, FrameGraph::CurrentOp());
		}
		// The verification: its GPU copy against its guest bytes, or its layers' contents.
		if (!resource.written && binding.image_id && FrameGraph::VerifyEnabled()) {
			textures.VerifyWithGuest(binding.image_id, FrameGraph::CurrentOp(),
			                         fmt::format("{} tex[{}]", stage, i));
		}
		// KYTY_DUMP_IMAGES: the image as this binding finds it, to a PNG (with
		// KYTY_DUMP_IMAGES_MIPS, only where no binding of this stage stores to it).
		const auto stage_writes = [&](const auto& id) {
			for (size_t j = 0; j < prepared.images.size() && j < program.info.images.size(); j++) {
				if (prepared.images[j].image_id == id && program.info.images[j].written) {
					return true;
				}
			}
			return false;
		};
		if (binding.image_id && ImageDump::Enabled() &&
		    ImageDump::WantedShader(program.shader_hash) &&
		    ImageDump::Wanted(info.data.address, info.extent.width, info.extent.height,
		                      info.resources.layers, info.IsDepth()) &&
		    !(ImageDump::AllMips() && stage_writes(binding.image_id))) {
			const auto capture = FrameGraph::CaptureNumber();
			(void)textures.DumpImage(
			    binding.image_id, capture,
			    fmt::format("cap{}_op{}_{}_t{}", capture, FrameGraph::CurrentOp(), stage, i),
			    fmt::format("FG {} {} tex[{}] {} of {:016x}", FrameGraph::CurrentOp(), stage, i,
			                resource.written ? "write" : "read", program.shader_hash));
		}
		// A small linear image a shader stores to, the kind the CPU reads back: shown after the
		// frame as guest memory holds it.
		if (resource.written && binding.desc.type == TextureCache::BindingType::Storage &&
		    info.tile_mode == Prospero::TileMode::kLinear && info.data.size <= 64u * 1024u) {
			FrameGraph::NoteReadback(info.data.address, info.data.size,
			                         textures.ReadsBackForCpu(info));
		}
	}
	for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
		const auto     r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const uint64_t bytes =
		    std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
		// As FindBuffers binds it: the mapped part only (an unused V# may hold anything, and the
		// memory tracker ends the emulator on a range outside guest memory).
		const uint64_t mapped = r.Base48() >= BufferCache::CACHING_PAGESIZE && bytes != 0
		                            ? Libs::LibKernel::Memory::TryClampRangeSize(r.Base48(), bytes)
		                            : 0;
		const bool     gpu    = mapped != 0 && buffers.HasGpuDirtyBytes(r.Base48(), mapped);
		const bool     cpu    = mapped != 0 && buffers.IsRegionCpuModified(r.Base48(), mapped);
		const auto     writers =
            mapped != 0 && gpu ? BufferCache::GpuWritersOf(r.Base48(), mapped) : std::string();
		// Round 90: and a hash of what a read-only one holds (FrameGraph::BytesHash).
		const auto hash = mapped != 0 && !program.info.buffers[i].written
		                      ? " hash " + FrameGraph::BytesHash(buffers, r.Base48(), mapped)
		                      : std::string();
		FrameGraph::Log("  %s buf[%zu] %s%s 0x%012" PRIx64 "+0x%" PRIx64 " (mapped 0x%" PRIx64
		                ") stride %u records %u fmt %u [%c%c] writers %s%s",
		                stage, i, program.info.buffers[i].written ? "write" : "read",
		                program.info.buffers[i].formatted ? " formatted" : "", r.Base48(), bytes,
		                mapped, r.Stride(), r.NumRecords(), r.RawFormat(), gpu ? 'G' : '-',
		                cpu ? 'C' : '-', writers.empty() ? "-" : writers.c_str(), hash.c_str());
		// Round 91: the first bytes of a small read-only buffer of large records (a draw's
		// instances: 240 bytes each for the crowds), as the draw reads them.
		if (mapped != 0 && mapped <= 16u * 1024u && r.Stride() >= 64u &&
		    !program.info.buffers[i].written && !gpu) {
			FrameGraph::LogBytes(fmt::format("{} buf[{}] bytes", stage, i).c_str(), r.Base48(),
			                     std::min<uint64_t>(mapped, 512u));
		}
		if (mapped != 0) {
			FrameGraph::NoteBuffer(r.Base48(), mapped);
			if (FrameGraph::DumpsShaderMemory(program.shader_hash)) {
				// Round 105: all of it, as it is after the frame (a crowd pass's instance list),
				// and for a dispatch, the GPU's copy as the dispatch finds it (a transient list a
				// later frame reuses is gone by the end of this one).
				FrameGraph::NoteChosenBuffer(r.Base48(), mapped);
				// Round 108: a draw's stages too (the GPU's copy of a table a pixel shader reads,
				// against the guest bytes after the frame).
				const bool compute = program.stage == ShaderType::Compute;
				// A draw's: once a capture for each range (a pass draws with the same table
				// hundreds of times).
				static uint32_t                                snapshot_capture = 0;
				static std::set<std::pair<uint64_t, uint64_t>> snapshot_ranges;
				if (snapshot_capture != FrameGraph::CaptureNumber()) {
					snapshot_capture = FrameGraph::CaptureNumber();
					snapshot_ranges.clear();
				}
				if (compute || (!program.info.buffers[i].written &&
				                snapshot_ranges.emplace(r.Base48(), mapped).second)) {
					buffers.SnapshotForDiagnostics(
					    r.Base48(), std::min(mapped, FrameGraph::ChosenBufferBytes()),
					    FrameGraph::CurrentOp(), fmt::format("{} buf[{}]", stage, i));
				}
				if (compute) {
					// Round 106: and what it wrote, right after it.
					if (program.info.buffers[i].written) {
						FrameGraph::NoteWrittenForSnapshot(
						    r.Base48(), std::min(mapped, FrameGraph::ChosenBufferBytes()),
						    fmt::format("cs buf[{}]", i));
					}
				}
			}
		}
		if (mapped != 0 && !gpu && !program.info.buffers[i].written &&
		    FrameGraph::VerifyEnabled()) {
			buffers.VerifyWithGuest(r.Base48(), mapped, FrameGraph::CurrentOp(),
			                        fmt::format("{} buf[{}]", stage, i));
		}
	}
	const auto& user = snapshot.user_data;
	for (size_t row = 0; row < user.size(); row += 16) {
		std::string line;
		for (size_t k = row; k < std::min(user.size(), row + 16); k++) {
			line += fmt::format(" {:08x}", user[k]);
		}
		FrameGraph::Log("  %s user s%zu:%s", stage, program.user_data_base + row, line.c_str());
	}
	// Pointers (a dword pair holding a mapped guest address) and V#s the shader may read its
	// constants through with scalar loads.
	for (size_t reg = 0; reg + 1 < user.size(); reg += 2) {
		const uint64_t address = user[reg] | (static_cast<uint64_t>(user[reg + 1]) << 32u);
		if ((user[reg] & 3u) == 0 && user[reg + 1] != 0 && user[reg + 1] <= 0xffu &&
		    Libs::LibKernel::Memory::TryClampRangeSize(address, 4) != 0) {
			FrameGraph::Log("  %s user pointer s[%zu:%zu] 0x%012" PRIx64 "%s", stage,
			                program.user_data_base + reg, program.user_data_base + reg + 1, address,
			                buffers.HasGpuDirtyBytes(address, FrameGraph::KeepBytes) ? " [G]" : "");
			FrameGraph::NoteBuffer(address, FrameGraph::KeepBytes);
			if (FrameGraph::DumpsShaderMemory(program.shader_hash)) {
				FrameGraph::LogBytes(fmt::format("{} memory 0x{:012x}", stage, address).c_str(),
				                     address, 0x400u);
			}
		}
	}
	for (size_t reg = 0; reg + 4 <= user.size(); reg += 4) {
		ShaderBufferResource r;
		r.fields[0] = user[reg];
		r.fields[1] = user[reg + 1];
		r.fields[2] = user[reg + 2];
		r.fields[3] = user[reg + 3];
		const uint64_t size =
		    r.Stride() != 0 ? static_cast<uint64_t>(r.Stride()) * r.NumRecords() : r.NumRecords();
		if (r.Base48() < 0x100000000ull || size == 0 || size > (1ull << 32u) ||
		    Libs::LibKernel::Memory::TryClampRangeSize(r.Base48(), 4) == 0) {
			continue;
		}
		FrameGraph::Log("  %s user V# s[%zu:%zu] 0x%012" PRIx64 "+0x%" PRIx64
		                " stride %u records %u",
		                stage, program.user_data_base + reg, program.user_data_base + reg + 3,
		                r.Base48(), size, r.Stride(), r.NumRecords());
		FrameGraph::NoteBuffer(r.Base48(), size);
	}
}

// Watch mode: a pixel shader among the slowest draws (GpuTiming::ProbesGridsOf): its light grids
// against their guest bytes now and then, as this draw will read them.
static void ProbeSlowShaderGrids(RenderContext& context, const ShaderStageRuntime& runtime,
                                 const PreparedBindings& prepared) {
	const auto& program  = *runtime.program;
	auto&       textures = context.GetTextureCache();
	for (size_t i = 0; i < prepared.images.size() && i < program.info.images.size(); i++) {
		const auto& binding = prepared.images[i];
		const auto& info    = binding.desc.info;
		if (program.info.images[i].written || !binding.image_id ||
		    info.pixel_format != vk::Format::eR32G32Uint || info.resources.layers <= 1 ||
		    info.data.size > 0x40000u || !GpuTiming::ClaimGridProbe(info.data.address)) {
			continue;
		}
		LOGF("GpuTiming: ps %016" PRIx64 " drew slowly before: its grid tex[%zu] 0x%012" PRIx64
		     " (%ux%u, %u layers) against its guest bytes in the \"FG 0 grid\" lines\n",
		     program.shader_hash, i, info.data.address, info.extent.width, info.extent.height,
		     info.resources.layers);
		textures.CompareWithGuest(binding.image_id, 0);
	}
}

// A pixel shader's light grid as GT7's lit shaders read them: a two-channel uint 2D array texture
// of up to 256 KB, sampled, not stored (cells: a light-list offset, packed light counts).
static bool IsLightGridBinding(const ShaderRecompiler::IR::ImageResource& resource,
                               const TextureBinding&                      binding) {
	const auto& info = binding.desc.info;
	return !resource.written && binding.image_id && info.pixel_format == vk::Format::eR32G32Uint &&
	       info.resources.layers > 1 && !info.data.Empty() && info.data.size <= 0x40000u &&
	       (resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray ||
	        resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D);
}

void RenderExecutor::DescribeLightGrid(const PreparedBindings& pixel,
                                       GpuTiming::DrawDetail&  detail) {
	auto& grid = detail.grid;
	grid       = {};
	if (pixel.runtime == nullptr || !*pixel.runtime) {
		return;
	}
	const auto& program  = *pixel.runtime->program;
	const auto& snapshot = *pixel.runtime->resources;
	// The user-data buffer: s[0:1], when the stage's user data start at s0.
	if (program.user_data_base == 0 && snapshot.user_data.size() >= 2) {
		const uint64_t pointer =
		    snapshot.user_data[0] | (static_cast<uint64_t>(snapshot.user_data[1]) << 32u);
		constexpr std::array<uint32_t, 3> Offsets {0x528u, 0x53cu, 0x6a4u};
		grid.words_read = pointer != 0;
		for (size_t i = 0; i < Offsets.size() && grid.words_read; i++) {
			grid.words_read = LibKernel::Memory::TryReadBacking(pointer + Offsets[i],
			                                                    &grid.words[i], sizeof(uint32_t));
		}
	}
	auto& textures = m_context.GetTextureCache();
	for (size_t i = 0; i < pixel.images.size() && i < program.info.images.size(); i++) {
		const auto& binding = pixel.images[i];
		if (!IsLightGridBinding(program.info.images[i], binding)) {
			continue;
		}
		// The image the view is of: the grid's own, or a larger one it lies in.
		const auto* image = textures.m_slot_images.try_get(binding.image_id);
		if (image == nullptr || image->info.pixel_format != vk::Format::eR32G32Uint ||
		    image->info.data.Empty() || image->info.resources.layers == 0 ||
		    image->info.tile_mode != Prospero::TileMode::kLinear) {
			continue;
		}
		const auto& info           = image->info;
		grid.address               = info.data.address;
		grid.width                 = info.extent.width;
		grid.height                = info.extent.height;
		grid.layers                = info.resources.layers;
		grid.base                  = binding.desc.view_info.base_layer;
		grid.count                 = binding.desc.view_info.layer_count;
		const uint64_t layer_bytes = info.data.size / info.resources.layers;
		const uint64_t row_texels  = std::max(info.pitch, info.extent.width);
		if (grid.base >= grid.layers || row_texels * info.extent.height * 8u > layer_bytes) {
			return;
		}
		const uint32_t layers = std::min(grid.count, grid.layers - grid.base);
		if (layer_bytes * layers > 0x100000u) {
			return;
		}
		thread_local std::vector<uint32_t> words;
		words.resize(static_cast<size_t>(layer_bytes * layers / sizeof(uint32_t)));
		if (words.empty() ||
		    !LibKernel::Memory::TryReadBacking(info.data.address + layer_bytes * grid.base,
		                                       words.data(), words.size() * sizeof(uint32_t))) {
			return;
		}
		grid.read = true;
		for (uint32_t layer = 0; layer < layers; layer++) {
			for (uint32_t y = 0; y < info.extent.height; y++) {
				for (uint32_t x = 0; x < info.extent.width; x++) {
					const uint64_t at = (layer * layer_bytes + (y * row_texels + x) * 8u) / 4u + 1u;
					const uint32_t packed = words[at];
					grid.most_low         = std::max(grid.most_low, packed & 0xffffu);
					grid.most_mid         = std::max(grid.most_mid, (packed >> 16u) & 0xffu);
					grid.most_top         = std::max(grid.most_top, packed >> 24u);
					grid.over += (packed & 0xffffu) > 256u ? 1u : 0u;
				}
			}
		}
		return;
	}
}

void RenderExecutor::CheckLightGridView(const ShaderStageRuntime& runtime, uint32_t index,
                                        const TextureBinding& binding) {
	auto&       textures = m_context.GetTextureCache();
	const auto* image    = textures.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.pixel_format != vk::Format::eR32G32Uint ||
	    image->info.data.Empty() || image->info.resources.layers == 0 ||
	    image->info.tile_mode != Prospero::TileMode::kLinear) {
		return;
	}
	const auto&    info        = image->info;
	const uint64_t layer_bytes = info.data.size / info.resources.layers;
	const uint32_t row         = std::max(info.pitch, info.extent.width);
	const uint32_t base        = binding.desc.view_info.base_layer;
	// The small grids only (the reflection cube faces' 8x8 cells): the screen's own grid never
	// held such cells, and its 70 KB a frame are not worth reading.
	if (info.extent.width * info.extent.height > 256u) {
		return;
	}
	if (layer_bytes == 0 || base >= info.resources.layers ||
	    binding.desc.view_info.layer_count == 0 ||
	    static_cast<uint64_t>(row) * info.extent.height * 8u > layer_bytes) {
		return;
	}
	const uint32_t layers =
	    std::min(binding.desc.view_info.layer_count, info.resources.layers - base);
	const uint64_t start = info.data.address + layer_bytes * base;
	const uint64_t size  = layer_bytes * layers;
	if (size > 0x40000u) {
		return;
	}
	// Once a guest submission for each place a view starts at (GPU thread only).
	struct View {
		uint32_t draw    = 0;
		uint64_t tsharp  = 0; // the T#'s address
		int64_t  first   = 0; // the view's first layer, counted from the T#'s address
		uint32_t layers  = 0;
		uint32_t garbage = 0;
	};
	struct Submission {
		uint64_t                     seq = 0;
		std::unordered_set<uint64_t> starts;
		std::vector<View>            views;
	};
	thread_local Submission seen;
	const auto              seq = g_guest_submission_seq.load(std::memory_order_relaxed);
	if (seen.seq != seq) {
		seen.seq = seq;
		seen.starts.clear();
		seen.views.clear();
	}
	if (!seen.starts.insert(start).second) {
		return;
	}
	// Not where the GPU wrote (the guest bytes there are older than the GPU's).
	thread_local std::vector<uint8_t> bytes;
	bytes.resize(static_cast<size_t>(size));
	if (!LibKernel::Memory::TryReadGpuCleanBacking(start, bytes.data(), size)) {
		return;
	}
	uint32_t garbage = 0;
	uint32_t most    = 0;
	for (uint32_t layer = 0; layer < layers; layer++) {
		for (uint32_t y = 0; y < info.extent.height; y++) {
			for (uint32_t x = 0; x < info.extent.width; x++) {
				uint32_t packed = 0;
				std::memcpy(&packed,
				            bytes.data() + layer * layer_bytes +
				                (static_cast<uint64_t>(y) * row + x) * 8u + 4u,
				            sizeof(packed));
				most = std::max(most, packed & 0xffffu);
				garbage += (packed & 0xffffu) > 1024u ? 1u : 0u;
			}
		}
	}
	Common::PerfStats::AddLightGridView(garbage);
	const auto* current = DrawPrep::Current();
	const auto  tsharp  = binding.desc.info.data.address;
	View        view;
	view.draw   = current != nullptr ? current->ordinal : 0u;
	view.tsharp = tsharp;
	view.first  = (static_cast<int64_t>(start) - static_cast<int64_t>(tsharp)) /
	             static_cast<int64_t>(layer_bytes);
	view.layers  = layers;
	view.garbage = garbage;
	if (seen.views.size() < 48) {
		seen.views.push_back(view);
	}
	if (garbage == 0) {
		return;
	}
	static std::atomic<uint64_t> found {0};
	const auto                   n = found.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n > 32 && n % 64 != 0) {
		return;
	}
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	std::string words;
	if (index < snapshot.images.size()) {
		const auto& descriptor = snapshot.images[index];
		for (uint32_t w = 0; w < descriptor.dword_count && w < 8u; w++) {
			words += fmt::format(" {:08x}", descriptor.dwords[w]);
		}
	}
	std::string user;
	for (size_t u = 0; u < snapshot.user_data.size() && u < 16u; u++) {
		user += fmt::format(" {:08x}", snapshot.user_data[u]);
	}
	// Where the T# came from: the first 8 KB behind each user-data pointer (an aligned pair of
	// user-data words that reads as a guest address), searched for its first four words; and who
	// wrote it there, when.
	std::string source;
	uint64_t    source_address = 0;
	if (index < snapshot.images.size() && snapshot.images[index].dword_count >= 4u) {
		const auto&                        d = snapshot.images[index].dwords;
		thread_local std::vector<uint32_t> window;
		window.resize(2048);
		for (size_t u = 0; u + 1 < snapshot.user_data.size() && source_address == 0; u += 2) {
			const uint64_t pointer =
			    snapshot.user_data[u] | (static_cast<uint64_t>(snapshot.user_data[u + 1]) << 32u);
			if (pointer < 0x100000u || (pointer & 3u) != 0 || pointer >= (uint64_t {1} << 40u) ||
			    !LibKernel::Memory::TryReadBacking(pointer, window.data(),
			                                       window.size() * sizeof(uint32_t))) {
				continue;
			}
			for (size_t w = 0; w + 4 <= window.size(); w++) {
				if (window[w] == d[0] && window[w + 1] == d[1] && window[w + 2] == d[2] &&
				    window[w + 3] == d[3]) {
					source_address = pointer + w * 4u;
					source = fmt::format("s[{}:{}] -> 0x{:x} +0x{:x}", program.user_data_base + u,
					                     program.user_data_base + u + 1, pointer, w * 4u);
					break;
				}
			}
		}
	}
	if (source_address == 0) {
		source = "not within 8 KB of a user-data pointer";
	}
	std::string others;
	for (const auto& other: seen.views) {
		others += fmt::format(" [{}: 0x{:x} {}+{}: {}]", other.draw, other.tsharp, other.first,
		                      other.layers, other.garbage);
	}
	LOGF("LightGrid view: submission %" PRIu64 ", draw %u, ps %016" PRIx64
	     " tex[%u]: T# 0x%012" PRIx64 " (%ux%u, %u layers) viewed at its layers %" PRId64
	     "+%u, in image 0x%012" PRIx64 " (%u layers): %u cells over 1024 lights (most %u; %" PRIu64
	     " so far) | T#:%s (read from %s) | user data:%s | the submission's light-grid views so "
	     "far (draw: T#, layers from the T#, cells over 1024):%s\n",
	     seq, view.draw, program.shader_hash, index, tsharp, binding.desc.info.extent.width,
	     binding.desc.info.extent.height, binding.desc.info.resources.layers, view.first, layers,
	     info.data.address, info.resources.layers, garbage, most, n, words.c_str(), source.c_str(),
	     user.c_str(), others.c_str());
	if (n <= 16) {
		LOGF("LightGrid view: 0x%012" PRIx64 "+0x%" PRIx64 " writes there in the last second, "
		     "newest first: %s\n",
		     start, size, LibKernel::Memory::DescribeJournal(start, size, 1000, 24).c_str());
		if (source_address != 0) {
			LOGF("LightGrid view: its T# at 0x%012" PRIx64 ": writes there in the last second, "
			     "newest first: %s\n",
			     source_address,
			     LibKernel::Memory::DescribeJournal(source_address, 32, 1000, 16).c_str());
		}
	}
}

// Round 106, diagnostics: KYTY_WATCH_CONSTANTS=<shader hash>:<user sgpr>:<offset>:<floats>
// (hex hash and offset, decimal sgpr and count; up to 4 watches, comma-separated) logs that many
// floats from the memory the pointer in that stage's user SGPR pair points at, plus the offset:
// a constant the CPU writes each frame (a colour correction a full-screen pass reads). A line
// when a value moved by more than 0.0005 since the last line, and at least every 120 frames.
namespace {
struct ConstantsWatch {
	uint64_t           hash   = 0;
	uint32_t           sgpr   = 0;
	uint32_t           offset = 0;
	uint32_t           count  = 0;
	std::vector<float> last;
	uint64_t           last_frame = 0;
	uint64_t           lines      = 0;
};
std::vector<ConstantsWatch>& ConstantsWatches() {
	static std::vector<ConstantsWatch> watches = [] {
		std::vector<ConstantsWatch> result;
		const char*                 text = std::getenv("KYTY_WATCH_CONSTANTS");
		while (text != nullptr && *text != '\0' && result.size() < 4) {
			ConstantsWatch watch;
			char*          end = nullptr;
			watch.hash         = std::strtoull(text, &end, 16);
			if (end == text || *end != ':') break;
			text       = end + 1;
			watch.sgpr = static_cast<uint32_t>(std::strtoul(text, &end, 10));
			if (end == text || *end != ':') break;
			text         = end + 1;
			watch.offset = static_cast<uint32_t>(std::strtoul(text, &end, 16));
			if (end == text || *end != ':') break;
			text = end + 1;
			watch.count =
			    std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(text, &end, 10)), 1u, 32u);
			if (end == text) break;
			result.push_back(std::move(watch));
			text = *end == ',' ? end + 1 : end;
		}
		for (const auto& watch: result) {
			LOGF("Constants watch: shader 0x%016" PRIx64 ", s[%u:%u] + 0x%x, %u floats\n",
			     watch.hash, watch.sgpr, watch.sgpr + 1, watch.offset, watch.count);
		}
		return result;
	}();
	return watches;
}

void WatchConstants(RenderContext& context, const ShaderStageRuntime& runtime) {
	auto& watches = ConstantsWatches();
	if (watches.empty() || runtime.program == nullptr || runtime.resources == nullptr) {
		return;
	}
	const auto& program = *runtime.program;
	for (auto& watch: watches) {
		if (watch.hash != program.shader_hash) {
			continue;
		}
		const auto& user = runtime.resources->user_data;
		if (watch.sgpr < program.user_data_base ||
		    watch.sgpr + 1 - program.user_data_base >= user.size()) {
			continue;
		}
		const auto         index   = watch.sgpr - program.user_data_base;
		const uint64_t     pointer = user[index] | (static_cast<uint64_t>(user[index + 1]) << 32u);
		std::vector<float> values(watch.count);
		if (pointer == 0 || !Libs::LibKernel::Memory::TryReadBacking(
		                        pointer + watch.offset, values.data(), values.size() * 4u)) {
			continue;
		}
		const uint64_t frame =
		    context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		if (frame == watch.last_frame && !watch.last.empty()) {
			continue;
		}
		bool moved = watch.last.size() != values.size();
		for (size_t i = 0; !moved && i < values.size(); i++) {
			moved = !(std::abs(values[i] - watch.last[i]) <= 0.0005f);
		}
		if (!moved && frame < watch.last_frame + 120u) {
			continue;
		}
		std::string line;
		for (const float value: values) {
			line += fmt::format(" {:.4f}", value);
		}
		if (watch.lines++ < 200000u) {
			LOGF("Constants watch (frame %" PRIu64 "): shader 0x%016" PRIx64 " 0x%012" PRIx64
			     "+0x%x:%s\n",
			     frame, watch.hash, pointer, watch.offset, line.c_str());
		}
		watch.last       = std::move(values);
		watch.last_frame = frame;
	}
}

// Round 121, diagnostics: KYTY_FIND_CONSTANTS=<shader hash>:<user sgpr>:<offset>:<bytes> (hex
// hash and offset, decimal sgpr and byte count, 8 to 64; up to 2, comma-separated). In each F9
// capture, at the first draw of that shader, the bytes at its user data pointer plus the offset
// (a colour correction the CPU writes each frame, say) are searched for in all of the guest's
// committed memory, 4-byte aligned: where else the CPU keeps them. Each place (up to 32) is logged
// with its mapping and the 512 bytes around it, as hex and as floats ("Find constants"). The first
// capture that finds places more than 64 MiB from the constants' own memory arms hardware write
// watches on up to four of them (Common::GuestWatch, Windows): the code that writes them is logged,
// with its bytes. KYTY_FIND_CONSTANTS_WATCH=0 leaves them unarmed.
struct FindConstantsItem {
	uint64_t hash         = 0;
	uint32_t sgpr         = 0;
	uint32_t offset       = 0;
	uint32_t bytes        = 0;
	uint32_t last_capture = 0;
};

std::vector<FindConstantsItem>& FindConstantsItems() {
	static std::vector<FindConstantsItem> items = [] {
		std::vector<FindConstantsItem> result;
		const char*                    text = std::getenv("KYTY_FIND_CONSTANTS");
		while (text != nullptr && *text != '\0' && result.size() < 2) {
			FindConstantsItem item;
			char*             end = nullptr;
			item.hash             = std::strtoull(text, &end, 16);
			if (end == text || *end != ':') break;
			text      = end + 1;
			item.sgpr = static_cast<uint32_t>(std::strtoul(text, &end, 10));
			if (end == text || *end != ':') break;
			text        = end + 1;
			item.offset = static_cast<uint32_t>(std::strtoul(text, &end, 16));
			if (end == text || *end != ':') break;
			text = end + 1;
			item.bytes =
			    std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(text, &end, 10)), 8u, 64u) &
			    ~3u;
			if (end == text) break;
			result.push_back(item);
			text = *end == ',' ? end + 1 : end;
		}
		for (const auto& item: result) {
			LOGF("Find constants: shader 0x%016" PRIx64 ", s[%u:%u] + 0x%x, %u bytes, searched "
			     "for in each F9 capture\n",
			     item.hash, item.sgpr, item.sgpr + 1, item.offset, item.bytes);
		}
		return result;
	}();
	return items;
}

// The words around a place, each as hex and as a float, 8 a line, every line naming the place (a
// search for "Find constants" keeps them all).
[[nodiscard]] std::string WordsAround(uint64_t address, uint64_t before, uint64_t after) {
	std::vector<uint32_t> words((before + after) / 4u);
	if (!Libs::LibKernel::Memory::TryReadBacking(address - before, words.data(),
	                                             words.size() * 4u)) {
		return fmt::format("Find constants 0x{:012x}: the bytes around it are unreadable\n",
		                   address);
	}
	std::string text;
	for (size_t i = 0; i < words.size(); i++) {
		if (i % 8u == 0) {
			const auto at = static_cast<int64_t>(i * 4u) - static_cast<int64_t>(before);
			text += fmt::format("{}Find constants 0x{:012x} {}0x{:03x}:", i == 0 ? "" : "\n",
			                    address, at < 0 ? "-" : "+", at < 0 ? -at : at);
		}
		float value = 0;
		std::memcpy(&value, &words[i], 4);
		text += fmt::format(" {:08x}={:.6g}", words[i], value);
	}
	return text + "\n";
}

void FindConstants(RenderContext& context, const ShaderStageRuntime& runtime) {
	auto& items = FindConstantsItems();
	if (items.empty() || !FrameGraph::Active() || runtime.program == nullptr ||
	    runtime.resources == nullptr) {
		return;
	}
	(void)context;
	const auto& program = *runtime.program;
	const auto  capture = FrameGraph::CaptureNumber();
	for (auto& item: items) {
		if (item.hash != program.shader_hash || item.last_capture == capture) {
			continue;
		}
		const auto& user = runtime.resources->user_data;
		if (item.sgpr < program.user_data_base ||
		    item.sgpr + 1 - program.user_data_base >= user.size()) {
			continue;
		}
		const auto     index   = item.sgpr - program.user_data_base;
		const uint64_t pointer = user[index] | (static_cast<uint64_t>(user[index + 1]) << 32u);
		const uint64_t own     = pointer + item.offset;
		std::vector<uint8_t> pattern(item.bytes);
		if (pointer == 0 ||
		    !Libs::LibKernel::Memory::TryReadBacking(own, pattern.data(), pattern.size())) {
			continue;
		}
		item.last_capture = capture;
		std::string values;
		for (size_t i = 0; i < pattern.size(); i += 4) {
			float value = 0;
			std::memcpy(&value, pattern.data() + i, 4);
			values += fmt::format(" {:.6g}", value);
		}
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		constexpr uint64_t                         From = uint64_t {1} << 32u;
		constexpr uint64_t                         To   = uint64_t {1} << 40u;
		if (!Libs::LibKernel::Memory::CommittedSpans(From, To - From, size_t {1} << 20u, &spans)) {
			LOGF("Find constants (capture %u): too many mappings to search\n", capture);
			continue;
		}
		const auto            start = std::chrono::steady_clock::now();
		constexpr uint64_t    Chunk = uint64_t {4} << 20u;
		const auto            n     = static_cast<uint64_t>(pattern.size());
		std::vector<uint8_t>  buffer(static_cast<size_t>(Chunk));
		std::vector<uint64_t> hits;
		uint64_t              scanned = 0;
		for (const auto& [address, size]: spans) {
			for (uint64_t done = 0; done + n <= size && hits.size() < 32;) {
				const auto part = std::min<uint64_t>(size - done, Chunk);
				if (Libs::LibKernel::Memory::TryReadBacking(address + done, buffer.data(), part)) {
					scanned += part;
					for (uint64_t i = 0; i + n <= part && hits.size() < 32; i += 4) {
						if (std::memcmp(buffer.data() + i, pattern.data(), 4) == 0 &&
						    std::memcmp(buffer.data() + i, pattern.data(), n) == 0) {
							const auto at = address + done + i;
							if (hits.empty() || hits.back() < at) {
								hits.push_back(at);
							}
						}
					}
				}
				// Overlap the next chunk by the pattern, less one word.
				done += part < size - done ? part - (n - 4) : part;
			}
		}
		const auto ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
		        .count();
		LOGF("Find constants (capture %u): shader 0x%016" PRIx64 " reads%s at 0x%012" PRIx64
		     "; %zu places hold them in %.1f GB of guest memory (%zu mappings, %.0f ms)\n",
		     capture, item.hash, values.c_str(), own, hits.size(),
		     static_cast<double>(scanned) / 1e9, spans.size(), ms);
		std::vector<uint64_t> distant;
		for (const auto hit: hits) {
			const auto distance = hit > own ? hit - own : own - hit;
			const bool is_far   = distance > (uint64_t {64} << 20u);
			if (is_far) {
				distant.push_back(hit);
			}
			LOGF("Find constants (capture %u): 0x%012" PRIx64 "%s (%s), the 512 bytes around it "
			     "as hex and float:\n%s",
			     capture, hit,
			     hit == own ? " (the constants themselves)"
			     : is_far   ? ""
			                : " (within 64 MiB of them)",
			     Libs::LibKernel::Memory::DescribeMappings(hit, 16).c_str(),
			     WordsAround(hit, 0x100, 0x100).c_str());
		}
		static bool       armed = false;
		static const bool watch = [] {
			const char* value = std::getenv("KYTY_FIND_CONSTANTS_WATCH");
			return value == nullptr || value[0] != '0';
		}();
		if (!armed && watch && !distant.empty()) {
			armed = true;
			distant.resize(std::min<size_t>(distant.size(), 4));
			(void)Common::GuestWatch::Arm(distant.data(), static_cast<uint32_t>(distant.size()),
			                              "KYTY_FIND_CONSTANTS");
		}
	}
}
} // namespace

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	const DiagShaderScope diag_shader_scope(program.shader_hash);
	// Diagnostics: KYTY_WATCH_SHADER=<hash>[,<hash>...] logs the buffer and texture descriptors
	// these shaders are bound with (first few uses of each; KYTY_WATCH_SHADER_EVERY=<n> adds every
	// n-th use, up to 8 more), with the shaders that last wrote each buffer.
	static const uint32_t watch_every = [] {
		const char* value = std::getenv("KYTY_WATCH_SHADER_EVERY");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	const auto                   watched_use = WatchedShaderUse(program.shader_hash);
	static std::atomic<uint32_t> frame_uses {0};
	const bool in_watched_frame =
	    watched_use != UINT32_MAX &&
	    InWatchedShaderFrame(program.shader_hash,
	                         m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
	                         frame_uses);
	if (watched_use < 4u ||
	    (watched_use != UINT32_MAX && watch_every != 0 && watched_use % watch_every == 0 &&
	     watched_use / watch_every <= 8u) ||
	    in_watched_frame) {
		LOGF("WatchShader 0x%016" PRIx64 " use=%u frame=%" PRIu64 " stage=%u buffers=%zu images=%zu\n",
		     program.shader_hash, watched_use,
		     m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
		     static_cast<uint32_t>(program.stage), program.info.buffers.size(),
		     program.info.images.size());
		for (size_t i = 0; i < program.info.buffers.size() && i < snapshot.buffers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			// Whether the GPU has written bytes of the buffer that the CPU copy lacks.
			const uint64_t bytes =
			    std::max<uint64_t>(r.Stride(), 1u) * static_cast<uint64_t>(r.NumRecords());
			auto&      cache = m_context.GetBufferCache();
			const bool gpu_written =
			    r.Base48() != 0 && bytes != 0 && cache.HasGpuDirtyBytes(r.Base48(), bytes);
			// The tracker ends the emulator on a range outside guest memory: the mapped part.
			const uint64_t mapped =
			    r.Base48() >= BufferCache::CACHING_PAGESIZE && bytes != 0
			        ? Libs::LibKernel::Memory::TryClampRangeSize(r.Base48(), bytes)
			        : 0;
			const bool cpu_modified = mapped != 0 && cache.IsRegionCpuModified(r.Base48(), mapped);
			const auto writers = r.Base48() != 0 && bytes != 0
			                         ? BufferCache::GpuWritersOf(r.Base48(), bytes)
			                         : std::string();
			LOGF("  buffer[%zu]: pc=0x%x formatted=%d written=%d base=0x%012" PRIx64
			     " stride=%u records=%u format=%u dst_sel=%u%u%u%u gpu_written=%d cpu_modified=%d"
			     " writers=%s\n",
			     i, program.info.buffers[i].first_use_pc, program.info.buffers[i].formatted ? 1 : 0,
			     program.info.buffers[i].written ? 1 : 0, r.Base48(), r.Stride(), r.NumRecords(),
			     r.RawFormat(), r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW(),
			     gpu_written ? 1 : 0, cpu_modified ? 1 : 0,
			     writers.empty() ? "-" : writers.c_str());
			// Small buffers (constants, tables) are listed in full, as guest memory holds them.
			if (r.Base48() != 0 && bytes != 0 && bytes <= 4096 && !gpu_written) {
				std::vector<uint32_t> words((bytes + 3) / 4);
				if (Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), bytes)) {
					for (size_t row = 0; row < words.size(); row += 8) {
						std::string line;
						for (size_t k = row; k < std::min(words.size(), row + 8); k++) {
							line += fmt::format(" {:08x}", words[k]);
						}
						LOGF("    +0x%03zx:%s\n", row * 4, line.c_str());
					}
				}
			}
		}
		for (size_t i = 0; i < program.info.images.size() && i < snapshot.images.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
			LOGF("  image[%zu]: bindless=%d format=%u dst_sel=%u%u%u%u type=%u extent=%ux%u\n", i,
			     program.info.images[i].bindless ? 1 : 0, static_cast<uint32_t>(r.Format()),
			     r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW(),
			     static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Width5()) + 1u,
			     static_cast<uint32_t>(r.Height5()) + 1u);
		}
		// User data, and the first bytes of every buffer a user-data V# points at: the shader
		// reads constants and heap indices from these with scalar loads.
		const auto& user = snapshot.user_data;
		for (size_t row = 0; row < user.size(); row += 8) {
			std::string line;
			for (size_t k = row; k < std::min(user.size(), row + 8); k++) {
				line += fmt::format(" {:08x}", user[k]);
			}
			LOGF("  user_data[s%zu]:%s\n", program.user_data_base + row, line.c_str());
		}
		for (size_t reg = 0; reg + 4 <= user.size(); reg += 4) {
			ShaderBufferResource r;
			r.fields[0]         = user[reg];
			r.fields[1]         = user[reg + 1];
			r.fields[2]         = user[reg + 2];
			r.fields[3]         = user[reg + 3];
			const uint64_t size = r.Stride() != 0 ? static_cast<uint64_t>(r.Stride()) * r.NumRecords()
			                                      : r.NumRecords();
			if (r.Base48() == 0 || size == 0) {
				continue;
			}
			std::vector<uint32_t> words(std::min<uint64_t>(size, 0x100u) / 4u);
			if (words.empty() || m_context.GetBufferCache().HasGpuDirtyBytes(r.Base48(), words.size() * 4u) ||
			    !Libs::LibKernel::Memory::TryReadBacking(r.Base48(), words.data(), words.size() * 4u)) {
				continue;
			}
			LOGF("  user V# s[%zu:%zu]: base=0x%012" PRIx64 " stride=%u records=%u\n",
			     program.user_data_base + reg, program.user_data_base + reg + 3, r.Base48(), r.Stride(),
			     r.NumRecords());
			LogWords(words.data(), words.size());
			std::lock_guard lock(g_watched_reread_mutex);
			g_watched_rereads.push_back(
			    {m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed) + 2u,
			     r.Base48(), words.size() * 4u});
		}
	}
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	const bool pixel     = program.stage == ShaderType::Pixel;
	const bool grids_off = pixel && Bisect::LightGridsOff();
	// KYTY_BISECT's hiz:off: a hull shader's texture sampled with unnormalized coordinates (GT7's
	// depth pyramid, against which it culls hidden patches) reads as zero, the far plane.
	const bool hiz_off = program.stage == ShaderType::TessellationControl && Bisect::HiZOff();
	const auto sampled_unnormalized = [&](uint32_t image) {
		for (const auto& pair: program.info.sampled_pairs) {
			if (pair.image != image || pair.sampler >= program.info.samplers.size()) {
				continue;
			}
			const auto index = program.info.samplers[pair.sampler].snapshot_index;
			if (index < snapshot.samplers.size() &&
			    DecodeNativeDescriptor<ShaderSamplerResource>(snapshot.samplers[index])
			        .ForceUnormCoords()) {
				return true;
			}
		}
		return false;
	};
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i], true);
		if (hiz_off &&
		    program.info.images[i].resource_class !=
		        ShaderRecompiler::IR::ImageResourceClass::Storage &&
		    sampled_unnormalized(i)) {
			auto desc = NullTextureDesc(program.info.images[i], TextureCache::BindingType::Texture);
			const auto id = m_context.GetTextureCache().FindImage(desc);
			binding       = {id, nullptr, std::move(desc)};
		}
		const bool grid    = pixel && IsLightGridBinding(program.info.images[i], binding);
		if (grid) {
			// Its later uploads are checked (TextureCache::CheckLightGridUpload), and what this
			// binding reads (CheckLightGridView).
			TextureCache::NoteLightGridShape(binding.desc.info);
			CheckLightGridView(runtime, i, binding);
		}
		if (grids_off && grid) {
			// KYTY_BISECT's grids:off: a null texture, whose cells read as zero lights.
			auto desc = NullTextureDesc(program.info.images[i], TextureCache::BindingType::Texture);
			const auto id = m_context.GetTextureCache().FindImage(desc);
			binding       = {id, nullptr, std::move(desc)};
		}
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	// KYTY_BISECT lit:big: whether a sampler samples a depth image 4096 or more texels wide. One
	// sampler can serve smaller depth images too (samplers are shared by their S#), which then
	// pass as well: logged, so a lit:big picture is read right.
	const auto samples_big_depth = [&](uint32_t sampler) {
		if (!Bisect::Active()) {
			return false;
		}
		auto& textures = m_context.GetTextureCache();
		bool  big      = false;
		bool  smaller  = false; // not "small": a macro in the Windows headers
		for (const auto& pair: program.info.sampled_pairs) {
			if (pair.sampler != sampler || pair.image >= prepared.images.size()) {
				continue;
			}
			const auto* image =
			    textures.m_slot_images.try_get(prepared.images[pair.image].image_id);
			if (image != nullptr && image->info.IsDepth()) {
				(image->info.extent.width >= 4096 ? big : smaller) = true;
			}
		}
		if (big && smaller) {
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.size() < 16 && logged.insert(program.shader_hash ^ sampler).second) {
				LOGF("Bisect: shader %016" PRIx64 " sampler %u samples a big and a smaller depth "
				     "image: with lit:big both pass\n",
				     program.shader_hash, sampler);
			}
		}
		return big;
	};
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index],
		    samples_big_depth(i), prepared.normalize_unnormalized));
	}
	for (const auto& use: snapshot.bindless_heaps) {
		if (m_bindless_surveyed.insert(use.base ^ (static_cast<uint64_t>(use.table_offset) << 48u))
		        .second) {
			SurveyBindlessHeap(use);
		}
	}
	PrepareBindlessHeaps(runtime, prepared);
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.bindings.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
	if (FrameGraph::Logging()) {
		LogFrameGraphStage(m_context, runtime, prepared);
	} else if (program.stage == ShaderType::Pixel &&
	           GpuTiming::ProbesGridsOf(program.shader_hash)) {
		ProbeSlowShaderGrids(m_context, runtime, prepared);
	}
	WatchConstants(m_context, runtime);
	FindConstants(m_context, runtime);
}

// KYTY_SPARSE_WRITES=0: a written buffer over memory with holes is bound up to its first hole.
static bool SparseWrites() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_WRITES");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// A written buffer binding over memory with holes in its mappings (sparse memory: the game maps
// 64 KB pages only where it needs them) is bound to the end of its last mapped byte, holes and
// all, instead of up to the first hole: the hardware's stores land in every mapped page (those to
// a hole are dropped, and in the emulator they stay in the cached buffer, never written back).
// Up to 1 GB and the device's storage-buffer range; `clamped` (up to the first hole) otherwise.
static uint64_t SparseWriteSize(RenderContext& context, uint64_t shader, uint64_t address,
                                uint64_t requested, uint64_t clamped) {
	if (!SparseWrites()) {
		return clamped;
	}
	const auto extent = Libs::LibKernel::Memory::CommittedExtent(address, requested);
	const auto limit  = std::min<uint64_t>(
        uint64_t {1} << 30u,
        context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange - 256u);
	if (extent <= clamped || extent > limit) {
		return clamped;
	}
	static std::mutex                              mutex;
	static std::set<std::pair<uint64_t, uint64_t>> logged;
	std::lock_guard                                lock(mutex);
	if (logged.size() < 32 && logged.insert({shader, address}).second) {
		LOGF("RenderExecutor: shader %016" PRIx64 " writes 0x%016" PRIx64 "+0x%" PRIx64
		     " over memory with holes: bound to its last mapped byte, 0x%" PRIx64
		     " (up to the first hole it was 0x%" PRIx64 ")\n",
		     shader, address, requested, extent, clamped);
	}
	return extent;
}

void RenderExecutor::FindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       cache    = m_context.GetBufferCache();

	prepared.buffer_sources.clear();
	const auto& layout = program.bindings;
	if (layout.memory_offset_count == 0) {
		return;
	}
	const auto& resources = layout.descriptors.front().resources;
	prepared.buffer_sources.reserve(resources.size());
	for (const auto resource: resources) {
		auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
		const auto address = descriptor.Base48();
		auto       requested_size = descriptor.GetSize();
		// Round 124, upstream fde0b98e (nmzik), ported by Franseven: OOB_SELECT=2 only tests
		// NumRecords != 0; a zero-stride fetch still needs its full payload when NumRecords is
		// smaller than it.
		if (requested_size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
		    resource < program.info.buffers.size() && !program.info.buffers[resource].scalar) {
			requested_size =
			    std::max<uint64_t>(requested_size, program.info.buffers[resource].max_byte_extent);
		}
		if (address != 0 && address < BufferCache::CACHING_PAGESIZE) {
			// A base in guest page 0 cannot be read on the hardware either; the shader must not
			// touch it, so it binds as null. Logged so the descriptor's origin can be traced.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("RenderExecutor: buffer descriptor %u in guest page 0: stage=%s words=[0x%08x "
				     "0x%08x 0x%08x 0x%08x] size=0x%016" PRIx64 "\n",
				     resource, ShaderStageResourceName(program.stage), descriptor.fields[0],
				     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
				     requested_size);
			}
		}
		if (address < BufferCache::CACHING_PAGESIZE || requested_size == 0) {
			prepared.buffer_sources.push_back({});
			continue;
		}
		uint64_t size = 0;
		{
			KYTY_PROFILER_BLOCK("FindBuffers::ClampRangeSize");
			size = Libs::LibKernel::Memory::TryClampRangeSize(address, requested_size);
		}
		if (size != requested_size && resource < program.info.buffers.size() &&
		    program.info.buffers[resource].written) {
			size = SparseWriteSize(m_context, program.shader_hash, address, requested_size, size);
		}
		{
			// A range past the device limit would EXIT in NativeStorageBuffer (a game binds one).
			const auto& graphics = m_context.GetGraphics();
			size                 = std::min(
                size,
                uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
                    (graphics.StorageMinAlignment() - 1));
		}
		if (size == 0) {
			// Nothing is mapped at the base yet. Streaming titles bind regions they back later (a
			// dispatch can precede the mapping by thousands of calls), and an unbacked region reads
			// zero and drops writes on the hardware; binding null gives the same.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("RenderExecutor: buffer descriptor %u outside mapped memory: stage=%s "
				     "base=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
				     resource, ShaderStageResourceName(program.stage), address, requested_size);
			}
			prepared.buffer_sources.push_back({});
			continue;
		}
		prepared.buffer_sources.push_back({address, size, cache.FindBuffer(address, size)});
	}
}

// The length in dwords that a binding's bounds checks compare with (round 84). Byte and halfword
// accesses index bytes and halfwords up to four or two times it, so a range rounded down to whole
// dwords hid the last bytes of one that ends inside a dword: the game's car glass reads a table
// of 17,370 bytes, its last patch read its last two as zero, and one corner of it rose half a
// metre above the roof. A read-only binding's length is rounded up: a dword read past the range
// is the driver's robust access (zero or bytes of the buffer), as the hardware's is zero. A
// written binding's stays rounded down, so that a dword store cannot reach past its range into
// the bytes after it. KYTY_BISECT's buflen:old rounds every binding down, as before.
static uint32_t BindingLengthDwords(uint64_t range, bool written) {
	const auto bytes = written || Bisect::BufferLengthOld() ? range : range + 3u;
	return static_cast<uint32_t>(bytes / sizeof(uint32_t));
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.memory_offset_count);
	const DiagShaderScope diag_shader_scope(program.shader_hash);

	prepared.buffers.clear();
	prepared.buffers.reserve(layout.memory_offset_count);
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	const bool write_tracking = layout.write_tracking;
	if (write_tracking) {
		const auto bitmap = m_context.GetBufferCache().WriteBitmapAddress();
		prepared.shader_data[layout.WriteTrackDword()]      = static_cast<uint32_t>(bitmap);
		prepared.shader_data[layout.WriteTrackDword() + 1u] = static_cast<uint32_t>(bitmap >> 32u);
	}
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (uint32_t i = 0; i < layout.memory_offset_count; i++) {
		const auto resource = layout.descriptors.front().resources[i];
		uint32_t buffer_offset = 0;
		uint64_t tracked_base  = 0;
		prepared.buffers.push_back(NativeStorageBuffer(
		    m_context, prepared.buffer_sources[i], program.info.buffers[resource],
		    resource < snapshot.buffer_write_extents.size() ? &snapshot.buffer_write_extents[resource]
		                                                    : nullptr,
		    buffer_offset, write_tracking && program.info.buffers[resource].write_tracked,
		    tracked_base));
		if (write_tracking) {
			const auto dword                     = layout.WriteTrackDword() + 2u + 2u * i;
			prepared.shader_data[dword]          = static_cast<uint32_t>(tracked_base);
			prepared.shader_data[dword + 1u]     = static_cast<uint32_t>(tracked_base >> 32u);
		}
		pack_memory_offset(i, buffer_offset);
		// The fallback allocation makes the Vulkan descriptor valid, but an empty guest
		// buffer still has no accessible elements. In particular, never let a null store
		// modify the shared fallback and affect subsequent null reads.
		prepared.shader_data[layout.BufferLengthDword() + i] =
		    prepared.buffer_sources[i].size == 0
		        ? 0u
		        : BindingLengthDwords(prepared.buffers.back().range,
		                              program.info.buffers[resource].written);
	}
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		if (prepared.bindless_patches.empty()) {
			prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
			FillFlatSlotsOnGpu(m_context, prepared.flattened_srt, snapshot.gpu_fills);
		} else {
			m_bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
			for (const auto& [offset, region, entries]: prepared.bindless_patches) {
				if (static_cast<size_t>(offset) + 1u < m_bindless_srt.size()) {
					m_bindless_srt[offset]      = region;
					m_bindless_srt[offset + 1u] = entries;
				}
				// The first 256 distinct patches are logged; after that no draw takes the lock.
				static std::atomic<bool>            logging_done = false;
				static std::mutex                   logged_mutex;
				static std::unordered_set<uint64_t> logged;
				if (logging_done.load(std::memory_order_relaxed)) {
					continue;
				}
				std::scoped_lock lock(logged_mutex);
				if (logged.size() >= 256) {
					logging_done.store(true, std::memory_order_relaxed);
				} else if (logged.insert(program.shader_hash ^ offset).second) {
					LOGF("Bindless patch: stage=%u hash=0x%016" PRIx64
					     " offset=%u region=%u entries=%u srt_words=%zu\n",
					     static_cast<uint32_t>(program.stage), program.shader_hash, offset, region,
					     entries, m_bindless_srt.size());
				}
			}
			prepared.flattened_srt = NativeUpload(m_context, m_bindless_srt);
			FillFlatSlotsOnGpu(m_context, prepared.flattened_srt, snapshot.gpu_fills);
		}
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	ReportTextureMemo();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Whether a sampler with unnormalized coordinates samples image i.
	const auto SampledUnnormalized = [&](uint32_t image) {
		for (const auto& pair: program.info.sampled_pairs) {
			if (pair.image != image || pair.sampler >= program.info.samplers.size()) {
				continue;
			}
			const auto index = program.info.samplers[pair.sampler].snapshot_index;
			if (index < snapshot.samplers.size() &&
			    DecodeNativeDescriptor<ShaderSamplerResource>(snapshot.samplers[index])
			        .ForceUnormCoords()) {
				return true;
			}
		}
		return false;
	};
	// The depth that passes the comparison of the samplers that sample image i with one: 1.0 for
	// LESS/LEQUAL, 0.0 for GREATER/GEQUAL. None without such a sampler, or for other functions
	// or samplers that disagree.
	const auto PassingDepth = [&](uint32_t image) -> std::optional<float> {
		std::optional<float> depth;
		for (const auto& pair: program.info.sampled_pairs) {
			if (pair.image != image || pair.sampler >= program.info.samplers.size()) {
				continue;
			}
			const auto& sampler = program.info.samplers[pair.sampler];
			if (!sampler.depth_compare) {
				continue;
			}
			if (sampler.bindless || sampler.snapshot_index >= snapshot.samplers.size()) {
				return std::nullopt;
			}
			float passing = 0.0f;
			switch (DecodeNativeDescriptor<ShaderSamplerResource>(
			            snapshot.samplers[sampler.snapshot_index])
			            .DepthCompareFunc()) {
				case 1: // LESS
				case 3: // LEQUAL
					passing = 1.0f;
					break;
				case 4: // GREATER
				case 6: // GEQUAL
					passing = 0.0f;
					break;
				default: return std::nullopt;
			}
			if (depth.has_value() && *depth != passing) {
				return std::nullopt;
			}
			depth = passing;
		}
		return depth;
	};
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				if (old_image->binding.is_target) {
					NoteBindlessStateChange(*old_image);
				}
				old_image->binding = {};
			}
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		// The texture memo: a binding the descriptor had before uses its view again while the
		// image is unchanged; otherwise it is looked up again from the start, as without the memo.
		// (Only for the plain view below: the views of the other branches are made from the
		// lookup, which a memo binding skipped.)
		const bool plain_view =
		    resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic &&
		    !(binding.desc.type != TextureCache::BindingType::Storage &&
		      binding.desc.view_info.level_count > 1 && SampledUnnormalized(i)) &&
		    !(binding.desc.type != TextureCache::BindingType::Storage && PassingDepth(i).has_value());
		if (binding.memo_view != nullptr) {
			if (plain_view && texture_cache.ReuseTextureBinding(binding.image_id)) {
				g_texture_memo_counts.reused++;
				binding.image_view = binding.memo_view;
				auto& image        = texture_cache.GetImage(binding.image_id);
				image.usage.texture = true;
				continue;
			}
			// The next binding of the descriptor is looked up as without the memo, and kept again
			// when the image may be kept.
			g_texture_memo_counts.refused++;
			ForgetTextureMemo(program.info.images[i], snapshot.images[i]);
			binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
			binding.mip_views.clear();
		}
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u);
			// Round 118: a T# the texture cache bound as a null texture (one with mips past its
			// last, ResolveTexture) has one level where the shader has a view for each of its
			// T#'s: the views past the binding's levels repeat its last one. Before, the emulator
			// ended here (issues 3 and 9, a race's 64x64 cube array of 7 mips whose descriptor
			// read with its upper half zero).
			const uint32_t levels = std::max(binding.desc.view_info.level_count, 1u);
			if (levels != resource.mip_count) {
				static std::atomic<uint32_t> reported {0};
				if (const auto count = reported.fetch_add(1, std::memory_order_relaxed) + 1u;
				    count <= 16u || std::has_single_bit(count)) {
					LOGF("Dynamic mips: %s shader %016" PRIx64 " tex[%u] 0x%012" PRIx64
					     " has %u levels bound where the shader selects among %u: the views past "
					     "them repeat its last level (%u so far)\n",
					     FrameGraphStage(program.stage), program.shader_hash, i,
					     binding.desc.info.data.address, levels, resource.mip_count, count);
				}
			}
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += std::min(mip, levels - 1u);
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else if (binding.desc.type != TextureCache::BindingType::Storage &&
		           binding.desc.view_info.level_count > 1 && SampledUnnormalized(i)) {
			// A compute shader's sampler with unnormalized coordinates (FORCE_UNNORMALIZED) gets a
			// view of the T#'s first mip, as a Vulkan sampler with unnormalizedCoordinates needs:
			// with all of them, the driver read others. The SDK's mip chain shaders sample the
			// level above this way, so every mip of GT7's reflection cubes came out one flat color
			// per face (2026-10-15). Graphics stages keep the whole view, as before: GT7's hull
			// shaders sample a depth pyramid this way at the mip they name, and with one mip its
			// cars came out with broken textures in third person (KYTY_BISECT mips:graphics put
			// them right, 2026-10-15).
			const bool one_mip =
			    program.stage == ShaderType::Compute && !Bisect::WholeUnnormalizedViews();
			auto       desc    = binding.desc;
			if (one_mip) {
				desc.view_info.level_count = 1;
				desc.view_info.min_lod     = 0;
			}
			binding.image_view = texture_cache.FindTexture(binding.image_id, desc);
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.size() < 32 && logged.insert(program.shader_hash ^ i).second) {
				LOGF("Unnormalized sampling: %s shader %016" PRIx64 " tex[%u] 0x%012" PRIx64
				     " %s mip %u (the T# has %u)\n",
				     FrameGraphStage(program.stage), program.shader_hash, i,
				     binding.desc.info.data.address, one_mip ? "viewed at" : "keeps every mip from",
				     binding.desc.view_info.base_level, binding.desc.view_info.level_count);
			}
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
			if (plain_view) {
				RecordTextureMemo(resource, snapshot.images[i], binding);
			}
		}
		// After FindTexture, which may refresh it from guest memory.
		if (binding.desc.type != TextureCache::BindingType::Storage) {
			if (const auto passing = PassingDepth(i); passing.has_value()) {
				texture_cache.ResetUnmappedDepth(binding.image_id, *passing);
			}
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
	}
}

bool DmaFaultReport() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DMA_FAULT_REPORT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void PrefaultDmaPointers(BufferCache& buffers, const ShaderStageRuntime& stage) {
	static const uint64_t span = [] {
		const char* value = std::getenv("KYTY_DMA_PREFAULT_KB");
		return (value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : 4ull)
		       << 10u;
	}();
	if (stage.program == nullptr || stage.resources == nullptr) {
		return;
	}
	const auto& program = *stage.program;
	const auto& user    = stage.resources->user_data;
	// Named in the fault lines; with KYTY_DMA_FAULT_REPORT=1, each use with its pointers.
	if (!DmaFaultReport()) {
		buffers.NoteDmaShader(program.shader_hash, {});
	} else {
		std::string label = fmt::format("{:016x}(", program.shader_hash);
		for (size_t reg = 0; reg + 1u < user.size(); reg += 2u) {
			label += fmt::format("{}{:08x}{:08x}", reg == 0 ? "" : ",", user[reg + 1u], user[reg]);
		}
		buffers.NoteDmaShader(program.shader_hash, label + ")");
	}
	if (span == 0) {
		return;
	}
	// A 64-bit SGPR pair starts at an even register; one a scalar load reads through is 4-byte
	// aligned and in one of the two windows the GPU sees of the guest's memory: below 1 TiB, and
	// the extended memory at 8 TiB (automatic memory).
	constexpr uint32_t ExtendedHigh =
	    static_cast<uint32_t>(LibKernel::Memory::kExtendedMemoryBase >> 32u);
	constexpr uint32_t ExtendedEnd = static_cast<uint32_t>(
	    (LibKernel::Memory::kExtendedMemoryBase + LibKernel::Memory::kExtendedMemorySize) >> 32u);
	for (size_t reg = program.user_data_base % 2u; reg + 1u < user.size(); reg += 2u) {
		const uint32_t low  = user[reg];
		const uint32_t high = user[reg + 1u];
		if ((low & 3u) != 0 || high == 0 ||
		    (high > 0xffu && (high < ExtendedHigh || high >= ExtendedEnd))) {
			continue;
		}
		const uint64_t address = low | (uint64_t {high} << 32u);
		const auto     made    = buffers.MakeResident(address, span);
		if (made == 0) {
			continue;
		}
		static std::atomic<uint64_t> total {0};
		const auto count = total.fetch_add(made, std::memory_order_relaxed) + made;
		if (count <= 32 || std::has_single_bit(count)) {
			LOGF("BufferCache: %u pages shader %016" PRIx64 " reads through user data s[%zu:%zu] "
			     "0x%012" PRIx64 " had no cached buffer: cached before it runs (%" PRIu64
			     " so far)\n",
			     made, program.shader_hash, program.user_data_base + reg,
			     program.user_data_base + reg + 1u, address, count);
		}
	}
}

// KYTY_TABLE: the bindings table (tableBindings.h).
uint32_t RenderExecutor::TableBindingsSlot(const PreparedBindings& prepared) const {
	for (uint32_t i = 0; i < m_graphics_bindings.vertex.size(); i++) {
		if (&prepared == &m_graphics_bindings.vertex[i]) {
			return i;
		}
	}
	EXIT_IF(!m_graphics_bindings.pixel || &prepared != &*m_graphics_bindings.pixel);
	return TableBindings::Slots - 1u;
}

// Diagnostics that PrepareBindings runs, or that change what it binds: their draws take the
// normal path.
static bool TableBindingsAllowed(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return !FrameGraph::Active() && !FrameGraph::Logging() && !Bisect::Active() &&
	       WatchedShaderHashes().empty() &&
	       !(program.stage == ShaderType::Pixel && GpuTiming::ProbesGridsOf(program.shader_hash));
}

bool RenderExecutor::TableBindingsMatch(PreparedBindings& prepared,
                                        const ShaderStageRuntime& runtime, bool normalize) {
	if (!m_binding_table) {
		m_binding_table = std::make_shared<TableBindings::Table>();
	}
	const auto slot  = TableBindingsSlot(prepared);
	auto&      entry = m_binding_table->slots[slot];
	m_binding_table->verify[slot].reset();
	if (!TableMode::ProgramHitFlag()) {
		TableMode::Add(TableMode::Counter::BindingMisses);
		TableMode::Add(TableMode::Counter::BindingMissProgram);
		return false;
	}
	const auto& snapshot = *runtime.resources;
	if (!entry.valid || entry.program != runtime.program || entry.normalize != normalize ||
	    prepared.runtime == nullptr || entry.images != snapshot.images ||
	    entry.samplers != snapshot.samplers || entry.buffers != snapshot.buffers ||
	    !snapshot.bindless_heaps.empty() || !snapshot.bindless_sampler_heaps.empty() ||
	    !TableBindingsAllowed(*runtime.program)) {
		TableMode::Add(TableMode::Counter::BindingMisses);
		TableMode::Add(TableMode::Counter::BindingMissKey);
		return false;
	}
	if (TableMode::Verifying()) {
		// The normal path prepares the stage; the bindings it would have reused are kept for
		// TableBindingsStore to compare, when they pass the reuse checks now.
		if (TableBindingsReuse(prepared, false)) {
			TableMode::Add(TableMode::Counter::BindingHits);
			m_binding_table->verify[slot] = prepared;
		} else {
			TableMode::Add(TableMode::Counter::BindingMisses);
			TableMode::Add(TableMode::Counter::BindingMissStale);
		}
		return false;
	}
	return true;
}

bool RenderExecutor::TableBindingsReuse(PreparedBindings& prepared, bool replay) {
	const auto& entry   = m_binding_table->slots[TableBindingsSlot(prepared)];
	auto&       texture = m_context.GetTextureCache();
	if (!entry.valid || entry.image_generation != texture.ImageSetGeneration()) {
		return false;
	}
	for (const auto& binding: prepared.images) {
		const auto* image = texture.m_slot_images.try_get(binding.image_id);
		if (image == nullptr || binding.image_view == nullptr) {
			return false;
		}
		if (!image->info.data.Empty() && !texture.ReuseTextureBinding(binding.image_id)) {
			return false;
		}
	}
	auto& buffers = m_context.GetBufferCache();
	if (prepared.buffers.size() != prepared.buffer_sources.size()) {
		return false;
	}
	for (size_t i = 0; i < prepared.buffer_sources.size(); i++) {
		const auto& source = prepared.buffer_sources[i];
		if (source.address < BufferCache::CACHING_PAGESIZE || source.size == 0) {
			continue;
		}
		if (!buffers.ReuseReadBinding(source.id, prepared.buffers[i].buffer, source.address,
		                              source.size)) {
			return false;
		}
	}
	if (replay) {
		// PrepareBindings' BindImage (its flags were cleared by the last draw's ResetBindings)
		// and the texture use FindTexture records; the layouts are CommitBindings' as always.
		for (const auto& binding: prepared.images) {
			BindImage(binding.image_id, false);
			auto& image = texture.GetImage(binding.image_id);
			if (!image.info.data.Empty()) {
				image.usage.texture = true;
			}
		}
	}
	return true;
}

void RenderExecutor::TableBindingsUpload(PreparedBindings& prepared) {
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	const auto& layout   = program.bindings;
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	// The user data words, as PrepareBindings copies them (the rest, buffer offsets and lengths,
	// is the same buffers' as before).
	for (size_t i = 0; i < layout.user_data_registers.size(); i++) {
		prepared.shader_data[i] = snapshot.user_data[layout.user_data_registers[i] -
		                                             program.user_data_base];
	}
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
		FillFlatSlotsOnGpu(m_context, prepared.flattened_srt, snapshot.gpu_fills);
	}
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

// Whether the normal path's bindings of a stage may be kept: read-only images and buffers the
// reuse checks can vouch for, no per-draw work PrepareBindings or RebindBuffers would redo.
static bool TableBindingsStorable(RenderContext& context, const PreparedBindings& prepared) {
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	const auto& layout   = program.bindings;
	if (!TableBindingsAllowed(program) || program.info.uses_dma || layout.write_tracking ||
	    !snapshot.bindless_heaps.empty() || !snapshot.bindless_sampler_heaps.empty() ||
	    !prepared.bindless_patches.empty() || !prepared.bindless_heaps.empty() ||
	    prepared.buffers.size() != prepared.buffer_sources.size() ||
	    prepared.images.size() != program.info.images.size()) {
		return false;
	}
	for (const auto& buffer: program.info.buffers) {
		if (buffer.written || buffer.formatted) {
			return false;
		}
	}
	for (size_t i = 0; i < prepared.images.size(); i++) {
		const auto& binding  = prepared.images[i];
		const auto& resource = program.info.images[i];
		if (resource.written || resource.bindless ||
		    resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage ||
		    binding.desc.type == TextureCache::BindingType::Storage ||
		    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic ||
		    !binding.mip_views.empty() || binding.image_view == nullptr) {
			return false;
		}
		if (program.stage == ShaderType::Pixel && IsLightGridBinding(resource, binding)) {
			return false;
		}
		// (The image itself: TableBindingsStore.)
	}
	auto& buffers = context.GetBufferCache();
	for (size_t i = 0; i < prepared.buffer_sources.size(); i++) {
		const auto& source = prepared.buffer_sources[i];
		if (source.address < BufferCache::CACHING_PAGESIZE || source.size == 0) {
			continue;
		}
		// Not the stream copy of CPU-dirty bytes (ObtainBuffer's small reads), nor a buffer
		// found again by ObtainBuffer: the cached buffer of the source, clean.
		if (!buffers.ReuseReadBinding(source.id, prepared.buffers[i].buffer, source.address,
		                              source.size)) {
			return false;
		}
	}
	return true;
}

// KYTY_TABLE=2: the first field in which the kept bindings differ from the normal path's.
static const char* TableBindingsDifference(const PreparedBindings& kept,
                                           const PreparedBindings& now) {
	if (kept.images.size() != now.images.size()) return "image count";
	for (size_t i = 0; i < kept.images.size(); i++) {
		const auto& a = kept.images[i];
		const auto& b = now.images[i];
		if (!(a.image_id == b.image_id)) return "image id";
		if (a.image_view != b.image_view) return "image view";
		if (a.desc.type != b.desc.type || !(a.desc.view_info == b.desc.view_info)) {
			return "image description";
		}
		if (a.mip_views != b.mip_views) return "mip views";
	}
	if (kept.samplers != now.samplers) return "samplers";
	if (kept.buffer_sources.size() != now.buffer_sources.size()) return "buffer count";
	for (size_t i = 0; i < kept.buffer_sources.size(); i++) {
		const auto& a = kept.buffer_sources[i];
		const auto& b = now.buffer_sources[i];
		if (a.address != b.address || a.size != b.size || !(a.id == b.id)) return "buffer source";
	}
	if (kept.buffers != now.buffers) return "buffer views";
	if (kept.gds != now.gds) return "gds";
	if (kept.normalize_unnormalized != now.normalize_unnormalized) return "normalization";
	// The kept shader data's user data words are this draw's after TableBindingsUpload; compare
	// the rest (buffer offsets and lengths).
	const auto& layout = now.runtime->program->bindings;
	if (kept.shader_data.size() != now.shader_data.size()) return "shader data size";
	for (size_t i = layout.user_data_registers.size(); i < now.shader_data.size(); i++) {
		if (kept.shader_data[i] != now.shader_data[i]) return "shader data";
	}
	return nullptr;
}

void RenderExecutor::TableBindingsStore(PreparedBindings& prepared, bool verify) {
	if (!m_binding_table) {
		m_binding_table = std::make_shared<TableBindings::Table>();
	}
	const auto slot  = TableBindingsSlot(prepared);
	auto&      entry = m_binding_table->slots[slot];
	auto&      kept  = m_binding_table->verify[slot];
	if (verify && kept) {
		TableMode::Add(TableMode::Counter::BindingVerifyChecks);
		if (const char* difference = TableBindingsDifference(*kept, prepared)) {
			TableMode::Add(TableMode::Counter::BindingVerifyMismatches);
			if (TableMode::VerifyLogAllowed()) {
				LOGF_UNMUTED("Table verify: bindings slot %u shader %016" PRIx64 ": %s differs (image set "
				     "generation %" PRIu64 " kept, %" PRIu64 " now; buffer set %" PRIu64
				     " kept, %" PRIu64 " now)\n",
				     slot, prepared.runtime->program->shader_hash, difference,
				     entry.image_generation, m_context.GetTextureCache().ImageSetGeneration(),
				     entry.buffer_generation, m_context.GetBufferCache().BufferSetGeneration());
			}
		}
	}
	kept.reset();
	auto&      texture  = m_context.GetTextureCache();
	const bool storable = TableBindingsStorable(m_context, prepared) &&
	                      std::ranges::all_of(prepared.images, [&](const TextureBinding& binding) {
		                      const auto* image = texture.m_slot_images.try_get(binding.image_id);
		                      return image != nullptr &&
		                             (image->info.data.Empty() ||
		                              texture.MayReuseTextureBinding(binding.image_id));
	                      });
	if (!storable) {
		entry.valid = false;
		TableMode::Add(TableMode::Counter::BindingNotStored);
		return;
	}
	const auto& snapshot = *prepared.runtime->resources;
	entry.valid          = true;
	entry.program        = prepared.runtime->program;
	entry.normalize      = prepared.normalize_unnormalized;
	entry.images         = snapshot.images;
	entry.samplers       = snapshot.samplers;
	entry.buffers        = snapshot.buffers;
	entry.image_generation  = m_context.GetTextureCache().ImageSetGeneration();
	entry.buffer_generation = m_context.GetBufferCache().BufferSetGeneration();
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors,
                                             uint32_t table_reused) {
	bool uses_dma = false;
	for (auto* stage: stages) {
		if (stage->runtime->program->info.uses_dma) {
			PrefaultDmaPointers(m_context.GetBufferCache(), *stage->runtime);
		}
	}
	{
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawFind);
		for (uint32_t i = 0; i < stages.size(); i++) {
			auto* stage = stages[i];
			if ((table_reused >> i) & 1u) {
				continue; // the slot's buffer sources, as they are
			}
			FindBuffers(*stage);
			uses_dma |= stage->runtime->program->info.uses_dma;
		}
	}
	if (uses_dma) {
		const GpuPhases::Scope phase(GpuPhases::Phase::DrawBda);
		m_context.PrepareBda();
	}
	GpuPhases::Scope images_phase(GpuPhases::Phase::DrawImages);
	for (uint32_t i = 0; i < stages.size(); i++) {
		auto* stage = stages[i];
		if ((table_reused >> i) & 1u) {
			if (TableBindingsReuse(*stage, true)) {
				TableMode::Add(TableMode::Counter::BindingHits);
				continue;
			}
			// The normal path, late: the stage is prepared here instead of with the others.
			TableMode::Add(TableMode::Counter::BindingMisses);
			TableMode::Add(TableMode::Counter::BindingMissStale);
			table_reused &= ~(1u << i);
			PrepareBindings(*stage->runtime, *stage);
			FindBuffers(*stage);
		}
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				if (old_image->binding.is_target) {
					NoteBindlessStateChange(*old_image);
				}
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	images_phase.End();
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	const GpuPhases::Scope buffers_phase(GpuPhases::Phase::DrawBuffers);
	for (uint32_t i = 0; i < stages.size(); i++) {
		if ((table_reused >> i) & 1u) {
			TableBindingsUpload(*stages[i]);
		} else {
			RebindBuffers(*stages[i]);
		}
	}
	if (TableMode::Enabled()) {
		for (uint32_t i = 0; i < stages.size(); i++) {
			if ((table_reused >> i) & 1u) {
				// Kept: valid against the generations as the draw leaves them.
				auto& entry = m_binding_table->slots[TableBindingsSlot(*stages[i])];
				entry.image_generation  = cache.ImageSetGeneration();
				entry.buffer_generation = m_context.GetBufferCache().BufferSetGeneration();
			} else {
				TableBindingsStore(*stages[i], TableMode::Verifying());
			}
		}
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	const auto vk_buffer    = buffer.Recorder();
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += program.bindings.descriptors.size();
		for (const auto& binding: program.bindings.descriptors) {
			descriptor_count += NativeDescriptorCount(binding);
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// Scalar reads the host specialized on must not overlap what this draw or dispatch writes.
	// The written ranges are gathered once, and a reader's reads are compared one by one only
	// against a range that meets their span: a per-read scan of every bound image and written
	// buffer cost the jungle 1.6 fps (11.1 against 12.7, warm).
	struct WrittenRange {
		uint64_t                                address = 0;
		uint64_t                                size    = 0;
		const ShaderRecompiler::IR::CompiledShaderInfo* writer = nullptr; // null: an image or target
		uint32_t                                resource = 0;
		uint64_t                                descriptor_address = 0;
		uint64_t                                descriptor_size    = 0;
		bool                                    bounded  = false;
	};
	// One draw or dispatch in 64 is checked (every one with KYTY_SPEC_READ_CHECK=1 or KYTY_BISECT's
	// speccheck:all): the check changes nothing, it reports (or stops on) what has never been
	// seen to happen, and it cost ~1 us a draw, 3 % of the GPU thread (round 68's sampler: the
	// read-by-read loop below was the hottest line of the whole thread).
	static const bool check_every = [] {
		const char* value = std::getenv("KYTY_SPEC_READ_CHECK");
		return value != nullptr && value[0] == '1';
	}();
	thread_local uint32_t check_counter = 0;
	const bool            check_spec_reads =
	    check_every || Bisect::SpecCheckAll() || (++check_counter & 63u) == 0u;
	GpuPhases::Scope spec_check_phase(GpuPhases::Phase::SpecCheck);
	if (!check_spec_reads) {
		spec_check_phase.Cancel();
	}
	std::vector<WrittenRange> written_ranges;
	bool                      written_gathered = false;
	for (const auto* reader:
	     check_spec_reads ? prepared_bindings : std::span<PreparedBindings* const> {}) {
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		if (!written_gathered) {
			written_gathered = true;
			for (const auto* writer: prepared_bindings) {
				if (writer->runtime->program->has_address_writes) {
					EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
				}
			}
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0) {
						written_ranges.push_back({.address = written.address, .size = written.size});
					}
				}
			}
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				const auto& extents = writer->runtime->resources->buffer_write_extents;
				for (uint32_t i = 0; i < writer->buffer_sources.size(); ++i) {
					const auto resource = program.bindings.descriptors.front().resources[i];
					if (!program.info.buffers[resource].written) continue;
					const auto& written = writer->buffer_sources[i];
					// The bytes the dispatch can store to, when its stores are bounded
					// (BufferWriteExtent): a title's heap-wide V# otherwise covers the whole
					// heap and every descriptor read in it.
					auto       written_address = written.address;
					auto       written_size    = written.size;
					const bool bounded = resource < extents.size() && extents[resource].valid;
					if (bounded) {
						written_address += extents[resource].begin;
						written_size = extents[resource].end > extents[resource].begin
						                   ? extents[resource].end - extents[resource].begin
						                   : 0;
					}
					if (written_size != 0) {
						written_ranges.push_back({.address            = written_address,
						                          .size               = written_size,
						                          .writer             = &program,
						                          .resource           = resource,
						                          .descriptor_address = written.address,
						                          .descriptor_size    = written.size,
						                          .bounded            = bounded});
					}
				}
			}
		}
		if (written_ranges.empty()) break;
		uint64_t span_begin = UINT64_MAX;
		uint64_t span_end   = 0;
		for (const auto [address, size]: reads) {
			span_begin = std::min(span_begin, address);
			span_end   = std::max(span_end, address + size);
		}
		for (const auto& written: written_ranges) {
			if (span_end <= span_begin ||
			    !ImageRangeOverlaps(span_begin, span_end - span_begin, written.address, written.size)) {
				continue;
			}
			for (const auto [address, size]: reads) {
				if (!ImageRangeOverlaps(address, size, written.address, written.size)) continue;
				if (written.writer == nullptr) {
					EXIT("scalar resource reads overlap an image or attachment write\n");
				}
				// Upstream (6409be28) exits here. Wolverine's CS 0x85a58319a7a75e48 reads a
				// branch input from a 256-byte buffer the same dispatch updates; the host
				// evaluates it from the state before the dispatch, as the walk before this
				// check did (and the game ran correctly). Reported once per shader pair.
				static std::mutex                   reported_mutex;
				static std::unordered_set<uint64_t> reported;
				const auto reader_hash = reader->runtime->program->shader_hash;
				std::scoped_lock lock(reported_mutex);
				if (reported.insert(reader_hash ^ (written.writer->shader_hash << 1u)).second) {
					LOGF("Warning: scalar resource reads overlap a shader buffer write: "
					     "reader 0x%016" PRIx64 " reads 0x%" PRIx64 "+0x%" PRIx64
					     ", writer 0x%016" PRIx64 " buffer %u writes 0x%" PRIx64 "+0x%" PRIx64
					     " (descriptor 0x%" PRIx64 "+0x%" PRIx64 ", extent %s)\n",
					     reader_hash, address, size, written.writer->shader_hash, written.resource,
					     written.address, written.size, written.descriptor_address,
					     written.descriptor_size, written.bounded ? "bounded" : "none");
				}
				break;
			}
		}
	}
	spec_check_phase.End();
	m_descriptor_buffers.clear();
	m_descriptor_images.clear();
	m_descriptor_writes.clear();
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	m_descriptor_writes.reserve(write_count);

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		// Bindless textures are sampled through set 1 with a read-only layout. A heap holds
		// thousands of resolved images; one whose images' states did not change since it was
		// last checked (g_bindless_state_generation) needs a look only at the images resolved
		// since. KYTY_BINDLESS_VERIFY=1 checks every image and stops if a skipped one needed it.
		static const bool verify = std::getenv("KYTY_BINDLESS_VERIFY") != nullptr;
		if (BindlessEngineEnabled() && !verify && !descriptors.bindless_heaps.empty()) {
			// Engine method, E5: only images settled since the last bindless draw, and pinned
			// images whose state changed (NoteBindlessStateChange), are made readable again; a
			// heap of thousands of images is not walked whenever any one of them changed.
			auto&      cache         = m_context.GetTextureCache();
			const auto make_readable = [&](ImageId id) {
				auto* image = cache.m_slot_images.try_get(id);
				if (image == nullptr || !image->registered || !image->bindless_pinned ||
				    image->binding.is_target || image->info.data.Empty()) {
					return;
				}
				const auto layout = image->info.IsDepth()
				                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                        : vk::ImageLayout::eShaderReadOnlyOptimal;
				if (image->backing.state.layout != layout) {
					const GuestImageUses guest_use;
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			};
			for (auto* heap: descriptors.bindless_heaps) {
				for (const auto id: heap->unchecked) {
					make_readable(id);
				}
				heap->unchecked.clear();
			}
			{
				std::scoped_lock lock {g_bindless_state_changes.mutex};
				m_bindless_changed.swap(g_bindless_state_changes.ids);
			}
			for (const auto id: m_bindless_changed) {
				make_readable(id);
			}
			m_bindless_changed.clear();
		}
		using HeapSpan = std::span<BindlessTable::Heap* const>;
		for (auto* heap: BindlessEngineEnabled() && !verify ? HeapSpan {}
		                                                    : HeapSpan {descriptors.bindless_heaps}) {
			const auto& resolved = heap->resolved;
			size_t      begin =
			    heap->checked_generation ==
			            g_bindless_state_generation.load(std::memory_order_relaxed) &&
			            heap->checked_count <= resolved.size()
			         ? heap->checked_count
			         : 0;
			if (verify) {
				begin = 0;
			}
			const auto skipped =
			    heap->checked_generation ==
			            g_bindless_state_generation.load(std::memory_order_relaxed) &&
			            heap->checked_count <= resolved.size()
			        ? heap->checked_count
			        : 0;
			for (size_t i = begin; i < resolved.size(); i++) {
				auto* image = m_context.GetTextureCache().m_slot_images.try_get(resolved[i]);
				if (image == nullptr || !image->registered || image->binding.is_target ||
				    image->info.data.Empty()) {
					continue;
				}
				const auto layout = image->info.IsDepth()
				                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                        : vk::ImageLayout::eShaderReadOnlyOptimal;
				if (image->backing.state.layout != layout) {
					if (verify && i < skipped) {
						EXIT("bindless: image %u of heap 0x%016" PRIx64
						     " changed layout without a state generation change\n",
						     static_cast<uint32_t>(i), heap->base);
					}
					const GuestImageUses guest_use;
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			}
			// After the transitions above, which bump the generation themselves.
			heap->checked_generation = g_bindless_state_generation.load(std::memory_order_relaxed);
			heap->checked_count      = resolved.size();
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			// The draw's or dispatch's own textures (images:lean, round 167).
			const GuestImageUses        guest_use;
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}

		m_image_occurrences.assign(descriptors.images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			vk::WriteDescriptorSet write {};
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				for (const auto resource: binding.resources) {
					m_descriptor_images.push_back(MakeImageInfo(
					    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
				}
			} else {
				switch (binding.kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != binding.resources.size());
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
							m_descriptor_buffers.push_back(view);
						}
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0,
						                                  bda_buffer->Size());
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (binding.kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (binding.kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (binding.kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						m_descriptor_buffers.push_back(*view);
						break;
					}
					case BindingKind::Samplers:
						for (const auto resource: binding.resources) {
							const auto sampler = descriptors.samplers.at(resource);
							EXIT_IF(sampler == nullptr);
							m_descriptor_images.emplace_back(sampler, nullptr,
							                                 vk::ImageLayout::eUndefined);
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		ScrubDeadDescriptorViews(m_descriptor_writes.data(),
		                         static_cast<uint32_t>(m_descriptor_writes.size()),
		                         "a draw's descriptors");
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			// The update moves to the recording thread with the bind: the set is fresh.
			vk_buffer.updateAndBindDescriptorSet(
			    m_context.GetGraphics().device, pipeline_bind_point, pipeline.pipeline_layout, set,
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data());
		}
	}
	if (pipeline.uses_bindless) {
		const auto set = m_context.GetGraphics().bindless_set;
		EXIT_IF(set == nullptr);
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 1, 1, &set, 0,
		                             nullptr);
	}
}

} // namespace Libs::Graphics
