#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/scalarReadCheck.h"
#include "graphics/host_gpu/renderer/pipeline/wolverineStores.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <cinttypes>
#include <algorithm>
#include <atomic>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

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
		case BindingKind::SharedMemory:
		case BindingKind::ProgramConstants: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

// PreparedBindings::Group::generation's counter.
std::atomic<uint64_t> g_group_generation {0};

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

// 1D textures are 2D images one texel high, so they share 2D surfaces at their address.
static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D:
		case Prospero::ImageType::kColor1DArray:
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

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset,
                    std::pair<uint64_t, uint64_t> written = {0, UINT64_MAX},
                    std::span<const std::pair<uint64_t, uint64_t>> written_ranges = {},
                    uint64_t* track_address = nullptr, Buffer** bound_buffer = nullptr) {
	buffer_offset = 0;
	if (bound_buffer != nullptr) *bound_buffer = nullptr;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	// Only the bytes the shader's stores can reach become GPU-written (ResourceSnapshot::buffer_writes).
	// Bounds past the binding mean a 32-bit offset wrap they cannot follow: all of it counts.
	const bool bounded        = written.second <= size;
	const auto written_offset = bounded ? written.first : 0u;
	const auto written_size   = bounded ? written.second - written.first : size;
	// Its stores have no bound but record the pages they write (BufferCache::TrackWrites).
	const bool tracked        = resource.tracked && !bounded && track_address != nullptr;
	auto [buffer, offset]     = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id,
	    written_offset, written_size, bounded ? written_ranges : std::span<const std::pair<uint64_t, uint64_t>> {},
	    tracked);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	if (tracked) {
		// The shader's indices start at the bound range's first byte.
		*track_address = context.GetBufferCache().TrackWrites(*buffer, address - adjustment, address, size);
	}
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (bound_buffer != nullptr) *bound_buffer = buffer;
	if (resource.written && written_size != 0) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address + written_offset, written_size);
	}
	return result;
}

// Makes a T# describe its texture from level `first` on, which is what the image heap holds of
// a streamed texture: that part of the chain lies at the start of the guest range, laid out as
// a texture of that size of its own would be. A texture the texture cache holds by level
// (TextureCache::HoldsLevels) keeps its whole chain and is read from level `first` on.
static void DropLevels(ShaderRecompiler::IR::DescriptorValue& value, uint32_t first, bool held) {
	auto& words = value.dwords;
	// shortcut: the fade of min LOD is kept to eighths of a level, so that a texture does not get
	// a view for each of the 256 steps of a level; follow it exactly when the steps show.
	auto min_lod = ((words[1] >> 8u) & 0xfffu) & ~0x1fu;
	if (held) {
		words[3] = (words[3] & ~(0xfu << 12u)) | (first << 12u); // base level
	} else if (first != 0u) {
		const auto width  = std::max(((((words[1] >> 30u) & 3u) | ((words[2] & 0xfffu) << 2u)) + 1u) >> first, 1u) - 1u;
		const auto height = std::max((((words[2] >> 14u) & 0x3fffu) + 1u) >> first, 1u) - 1u;
		words[1]          = (words[1] & 0x3fffffffu) | ((width & 3u) << 30u);
		words[2]          = (words[2] & ~(0xfffu | (0x3fffu << 14u))) | (width >> 2u) | (height << 14u);
		words[3] -= first << 16u; // last level
		words[5] -= first << 4u;  // highest level
		min_lod -= first << 8u;
	}
	words[1] = (words[1] & ~(0xfffu << 8u)) | (min_lod << 8u);
}

// The first level of a 2D texture's mip tail, its level count when it has none.
static uint32_t FirstTailLevel(const ShaderTextureResource& texture) {
	const TileSurfaceDescription description {texture.Format(), texture.TileMode(), TileSurfaceDimension::Dim2D,
	                                          texture.Width5() + 1u, texture.Height5() + 1u, 1u,
	                                          texture.LastLevel() + 1u, 1u};
	TileSurfaceLayout layout {};
	return texture.TileMode() != Prospero::TileMode::kLinear && TileGetTiledTextureLayout(description, layout)
	           ? layout.first_tail_level
	           : description.levels;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state and the
// mip-stats counter only counts sampling.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 1u << 29u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (uint32_t {descriptor.MipStatsCntEn()} << 25u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 || descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 && descriptor.fields[6] == 0 &&
		       descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	// Bits 0-7 name the mip-stats counter. A clear COMPRESSION_EN disables the metadata and its
	// settings.
	const auto         metadata_control = descriptor.fields[6] & 0x00ffff00u;
	constexpr uint32_t pipe_aligned     = 1u << 19u;
	if (!descriptor.MetaCompress()) {
		return true;
	}
	constexpr uint32_t htile_control = pipe_aligned | 1u << 21u;
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
	const bool is_2d_msaa =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    descriptor.Type() == Prospero::ImageType::kColor2DMsaa && descriptor.Depth() == 0 &&
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
	// A 4.8 fixed-point MIN_LOD at or below BASE_LEVEL clamps no level of the view.
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d || is_2d_msaa) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() <= descriptor.BaseLevel() * 256u && supported_swizzle &&
	       descriptor.BCSwizzle() == 0 && !descriptor.MsaaDepth();
}

bool ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size, bool fatal) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_storage      = uint_resource &&
	    (ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid ||
	     ShaderRecompiler::IR::PackedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid);
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return true;
	}
	if (!fatal) {
		return false;
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
	return false;
}

// sRGB views decode before filtering; a sampler that skips degamma reads through a UNorm view.
static vk::Format SampledViewFormat(const GraphicContext& graphics, Prospero::BufferFormat format,
                                    vk::Format pixel_format, bool skip_degamma) {
	if (skip_degamma) return SrgbUnormViewFormat(pixel_format);
	switch (format) {
		case Prospero::BufferFormat::k8Srgb:
			return graphics.r8_srgb_views ? vk::Format::eR8Srgb : pixel_format;
		case Prospero::BufferFormat::k8_8Srgb:
			return graphics.r8g8_srgb_views ? vk::Format::eR8G8Srgb : pixel_format;
		default: return pixel_format;
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
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
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

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical = info;
	// A view entirely past MAX_MIP addresses its own chain starting at its base mip (PPSA02110
	// compresses mip 1 through a T# that declares one level).
	if (base_level >= physical_levels) {
		const auto mip  = info.MipExtent(base_level);
		physical.extent = {mip.width, mip.height, info.IsVolume() ? mip.depth : 1u};
		physical_levels = view_levels;
	}
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		const auto& left  = physical.mip_layout[target];
		const auto& right = info.mip_layout[source];
		return left.offset == right.offset && left.size == right.size &&
		       left.pitch == right.pitch && left.height == right.height &&
		       left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
		       left.surface_z == right.surface_z &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

// The image heap's null image: heap images are sampled, never written, atomic or compared.
static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = vk::ImageUsageFlagBits::eSampled;
	desc.type                 = TextureCache::BindingType::Texture;
	// A null entry of a table takes the dimension of the table's other entries, so its view
	// has to be of the type the shader declares for them. A cube is sampled as a 2D array.
	using Dimension = ShaderRecompiler::Decoder::ImageDimension;
	switch (resource.dimension) {
		case Dimension::Dim3D:
			desc.info.type      = Prospero::ImageType::kColor3D;
			desc.view_info.type = vk::ImageViewType::e3D;
			break;
		// 1D images are 2D images one texel high.
		case Dimension::Dim1DArray:
		case Dimension::Dim2DArray: desc.view_info.type = vk::ImageViewType::e2DArray; break;
		default: break;
	}
	return desc;
}

RenderExecutor::TextureMemoResource::TextureMemoResource(const ShaderRecompiler::IR::ImageResource& resource)
    : resource_class(resource.resource_class), numeric_class(resource.numeric_class), dimension(resource.dimension),
      mip_mode(resource.mip_mode), mip_count(resource.mip_count), conversion_format(resource.conversion_format),
      shader_swizzle(resource.shader_swizzle), indirect_root(resource.indirect_root),
      indirect_mapping_offset(resource.indirect_mapping_offset),
      indirect_search_iterations(resource.indirect_search_iterations), indirect_capacity(resource.indirect_capacity),
      read(resource.read), written(resource.written), atomic(resource.atomic), atomic64(resource.atomic64),
      unfiltered_access_only(resource.unfiltered_access_only), sample_only(resource.sample_only),
      one_layer_safe(resource.one_layer_safe), base_level_only(resource.base_level_only),
      depth_compare(resource.depth_compare), cube(resource.cube),
      r128(resource.r128), heap(resource.heap), guest_heap(resource.guest_heap) {}

uint64_t RenderExecutor::TextureMemoHash(const ShaderRecompiler::IR::ImageResource&   resource,
                                         const ShaderRecompiler::IR::DescriptorValue& value, bool create) {
	uint64_t hash = 0xcbf29ce484222325ull ^ (uint64_t {value.dword_count} << 1u) ^ uint64_t {create};
	for (const auto dword: value.dwords) hash = (hash ^ dword) * 0x100000001b3ull;
	hash ^= (static_cast<uint64_t>(resource.resource_class) << 8u) ^ (static_cast<uint64_t>(resource.dimension) << 16u) ^
	        (static_cast<uint64_t>(resource.numeric_class) << 24u) ^ (uint64_t {resource.written} << 32u) ^
	        (uint64_t {resource.guest_heap} << 33u) ^ (uint64_t {resource.depth_compare} << 34u);
	hash ^= hash >> 33u;
	hash *= 0xff51afd7ed558ccdull;
	hash ^= hash >> 33u;
	return hash;
}

// The texture cache's image changes as Common::ResolveGroup takes them.
struct ImageChanges {
	TextureCache& cache;
	[[nodiscard]] uint64_t Clock() const { return cache.ImageChangeClock(); }
	template <typename Visit>
	[[nodiscard]] bool Since(uint64_t clock, Visit&& visit) const {
		return cache.ImageChangesSince(clock, std::forward<Visit>(visit));
	}
};

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              bool create, Common::GroupResult* outcome) {
	// A storage candidate binds as null over a target of the draw (HasTargetIn), and a group resolves its thousands
	// of candidates whole after any change to the set of images (group reuse, PrepareBindings): remembering them cost
	// more at Marvel's Wolverine's fight than it saved. They resolve every time.
	if (resource.written && resource.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage) {
		auto binding     = ResolveTextureFully(resource, value, create, outcome, nullptr);
		m_resolved_entry = nullptr;
		return binding;
	}
	// Ryujinx's texture bindings keep the last texture per binding while it is not invalidated, and sync its memory
	// on every use (TextureBindingsManager); here a binding is kept while its image is not (no image registered or
	// unregistered over its pages, axylpup's PREPARED_LOCAL idea), and what the lookup did besides finding the image
	// is done again on every use (TextureCache::RepeatLookup).
	auto&                     texture_cache = m_context.GetTextureCache();
	const TextureMemoResource key_resource(resource);
	const auto                hash  = TextureMemoHash(resource, value, create);
	auto*                     entry = m_texture_memo.Find(hash, [&](const auto& stored) {
        return stored.key.create == create && stored.key.value == value && stored.key.resource == key_resource;
    });
	if (entry != nullptr) {
		const auto& memo = entry->value;
		const auto& desc = memo.binding.desc;
		// In the order ResolveTextureFully checks them. The touch of the image after its lookup (GetImage) is the
		// lookup's own again: LruCache::Touch keeps an item touched at the same tick where it is. The binding's desc
		// is the lookup's but for a block image's one-level view (base_level_only), whose level count and min LOD
		// the color-clear materialization does not read.
		if ((memo.mapped_size == 0 || m_context.IsMapped(desc.info.data.address, memo.mapped_size)) &&
		    (!memo.lookup || texture_cache.RepeatLookup(memo.lookup, memo.stamp, desc, memo.metadata_base_layer))) {
			if (outcome != nullptr) *outcome = {memo.outcome};
			m_resolved_entry = entry;
			return memo.binding;
		}
	}
	TextureMemoValue made;
	auto             binding = ResolveTextureFully(resource, value, create, outcome, &made);
	m_resolved_entry = nullptr;
	if (made.cacheable) {
		if (entry == nullptr) entry = &m_texture_memo.Store(hash); // a stale entry of this key is filled again
		entry->key           = {key_resource, value, create};
		entry->value         = std::move(made);
		entry->value.binding = binding;
		entry->value.fill    = ++m_texture_memo_fills;
		m_resolved_entry     = entry;
	} else if (entry != nullptr) {
		entry->valid = false;
	}
	return binding;
}

TextureBinding RenderExecutor::ResolveTextureFully(const ShaderRecompiler::IR::ImageResource&   resource,
                                                   const ShaderRecompiler::IR::DescriptorValue& value, bool create,
                                                   Common::GroupResult* outcome, TextureMemoValue* made) {
	if (outcome != nullptr) *outcome = {};
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	// The image heap's slot 0, what a heap read of a slot the host has not filled falls back to,
	// is the 1x1 R32 null image: (0,0,0,1). With a native null descriptor's (0,0,0,0) Marvel's
	// Wolverine's title lost its depth of field and exposure: it reads heap slots that are empty.
	// Any other null T# is a native null descriptor (reads zero, writes dropped).
	using NumericClass = Prospero::TextureNumericClass;
	if (descriptor.IsNull() && (!resource.guest_heap || storage ||
	                            (resource.numeric_class != NumericClass::Float &&
	                             resource.numeric_class != NumericClass::Uint &&
	                             resource.numeric_class != NumericClass::Sint))) {
		if (outcome != nullptr) outcome->outcome = Common::GroupOutcome::Fixed;
		if (made != nullptr) {
			made->outcome   = Common::GroupOutcome::Fixed;
			made->cacheable = true;
		}
		return {.layout = vk::ImageLayout::eGeneral};
	}
	// Image heap resolves do not follow the draws that sample them and storage access is not
	// sampling: neither reports samples.
	if (m_lod_stats && descriptor.MipStatsCntEn() && !resource.heap && !storage) {
		auto&         counter = m_mip_stats[descriptor.MipStatsCntId()];
		const uint8_t base    = descriptor.BaseLevel();
		counter.finest_level  = std::min(counter.finest_level, base);
		if (descriptor.MinLod() > base << 8u) {
			counter.clamped = true;
		} else {
			counter.unclamped = true;
		}
	}
	if (descriptor.IsNull()) {
		auto       desc  = NullTextureDesc(resource);
		uint64_t   stamp = 0;
		const auto id    = texture_cache.FindImage(desc, false, true, made != nullptr ? &stamp : nullptr);
		if (made != nullptr) {
			made->lookup    = id;
			made->stamp     = stamp;
			made->cacheable = stamp != 0u;
		}
		return {id, nullptr, std::move(desc)};
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
	auto view_levels = multisampled || single_storage_mip
	                       ? 1u
	                       : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource)
	                                ? Prospero::BufferFormat::k16UNorm
	                                : descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	auto          image_layers = layered ? depth : 1u;
	// A cube holds six faces even when DEPTH, its last face slice, ends before them (Teardown's
	// sky cube has DEPTH 0).
	if (descriptor.Type() == Prospero::ImageType::kCube) {
		image_layers = std::max(image_layers, 6u);
	}
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	auto raw_storage_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	if (raw_storage_format == Prospero::BufferFormat::kInvalid) {
		raw_storage_format = ShaderRecompiler::IR::PackedStorageFormat(resource, format);
	}
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && resource.atomic ? vk::Format::eR32Uint
	                                 : raw_storage_format != Prospero::BufferFormat::kInvalid
	                                     ? VulkanFormat(raw_storage_format)
	                                     : SrgbUnormViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, 0};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		desc.info.metadata.kind = ImageMetadataKind::Dcc;
	}
	if (levels > physical_levels) {
		if (!ResolveTextureMipView(desc.info, physical_levels,
		                           !resource.r128 && descriptor.MetaCompress(), view_levels,
		                           view_base)) {
			// The layout stores no mip past MAX_MIP here, so a view starting in the chain ends at
			// MAX_MIP; dynamic-mip shaders still count mips up to LAST_LEVEL.
			if (base_level >= physical_levels ||
			    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
				EXIT("unsupported texture mip view changes physical layout: base=%u last=%u "
				     "max=%u extent=%ux%ux%u tile=%u\n",
				     base_level, last_level, max_mip, width, height, depth,
				     static_cast<uint32_t>(tile));
			}
			view_levels                = physical_levels - base_level;
			desc.info.resources.levels = physical_levels;
			desc.info.data.size        = 0;
			desc.info.UpdateSize();
		}
	} else {
		desc.info.UpdateSize();
	}
	const auto alignment = desc.info.DataAlignment();
	EXIT_NOT_IMPLEMENTED(desc.info.data.size == 0 || alignment == 0 ||
	                     (address & (static_cast<uint64_t>(alignment) - 1u)) != 0);
	// Image heaps keep the descriptors of unmapped textures, which PS5 never samples.
	if (resource.heap && !m_context.IsMapped(address, desc.info.data.size)) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	if (storage) {
		// A storage image picked out of a table by key binds every entry the key can reach (for a
		// material slot, that field of every material's block), and the shader writes one of them.
		// An entry no image store can write is another material's texture: a block-compressed one
		// (stores do not encode blocks), one the storage path does not take, like the streamed
		// BC1 texture with min LOD 1.0 among Marvel's Wolverine's particle heat-map slots, or one
		// over a target of this draw (no draw writes its own targets through a storage image; that
		// heat map's slots include the R32 view of the depth buffer its particles test against).
		// It binds as nothing. So does an entry without an image of its own backing: making or
		// replacing images for every entry (2,078 textures, 4 GB more video memory in that game's
		// first fight) costs more than any one store.
		// shortcut: a store whose target nothing else has made yet loses its writes; bound the
		// keys to the draw's own records in the materialization, then make images again.
		const bool candidate = resource.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage;
		const bool fixed     = candidate && (block_bytes != 0 ||
		                                     !ValidateStorageTexture(resource, descriptor, desc.info.data.size, false));
		const bool target    = candidate && !fixed && texture_cache.HasTargetIn(address, desc.info.data.size);
		if (candidate && outcome != nullptr) {
			outcome->address = address;
			outcome->end     = address + desc.info.data.size;
		}
		if (candidate && (fixed || target || !texture_cache.HasImageFor(desc.info))) {
			if (outcome != nullptr && !target) {
				outcome->outcome = fixed ? Common::GroupOutcome::Fixed : Common::GroupOutcome::Absent;
			}
			static std::atomic_flag logged;
			if (!logged.test_and_set()) {
				LOGF("storage image candidate bound as null: dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
				     descriptor.fields[0], descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
				     descriptor.fields[4], descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
			}
			return {.layout = vk::ImageLayout::eGeneral};
		}
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	desc.heap = resource.heap;

	const auto metadata_base_layer = desc.view_info.base_layer;
	uint64_t   stamp               = 0;
	auto       id = texture_cache.FindImage(desc, shader_conversion, create, made != nullptr ? &stamp : nullptr);
	if (!id) {
		// An image heap descriptor another extent at the address holds: the heap retries once an image is
		// created (95bbc9c65).
		if (desc.heap) return {.desc = std::move(desc), .layout = vk::ImageLayout::eGeneral};
		return {};
	}
	if (made != nullptr) {
		made->lookup              = id;
		made->stamp               = stamp;
		made->metadata_base_layer = metadata_base_layer;
		made->mapped_size         = resource.heap ? desc.info.data.size : 0u;
		made->cacheable           = stamp != 0u;
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		// Set after FindImage, whose metadata clears need the attachment-capable format.
		desc.view_info.format = SampledViewFormat(m_context.GetGraphics(), format, pixel_format,
		                                          resource.skip_degamma);
		(void)SelectSampledColorView(image->info.pixel_format, desc.view_info.format,
		                             descriptor.DstSelXYZW());
	}
	// An uncompressed view of a block-compressed image holds one level (VUID-VkImageViewCreateInfo-image-07072):
	// its texels are the blocks of that level. A shader that reads only the view's base level, as Marvel's
	// Wolverine's raw R32G32B32A32 loads of its BC6H probe cubes do, reads the same from a view of that level.
	// A fetch takes the whole levels of the min LOD, so one under a level moves nothing, and in a view of one
	// level it would point past that level (VUID-VkImageViewMinLodCreateInfoEXT-minLod-06456).
	if (image->info.IsBlock() && !ImageViewOps::IsBlockFormat(desc.view_info.format) &&
	    desc.view_info.level_count > 1 && resource.base_level_only && desc.view_info.min_lod < 256u) {
		desc.view_info.level_count = 1;
		desc.view_info.min_lod     = 0;
	}
	return {id, nullptr, std::move(desc)};
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
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

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.binding.is_bound) {
		if (image.binding.shader_write != storage && !image.binding.force_general) m_force_general_images.push_back(id);
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
	m_target_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		m_context.GetTextureCache().m_slot_images[id].binding = {};
	}
	m_bound_images.clear();
	m_target_images.clear();
	m_force_general_images.clear();
}

namespace {

// A program with a group and one without take turns on the same slot within a pass, and each turn threw the group's
// results away: Marvel's Wolverine's Trask outpost resolved all 1,218 candidates on most such draws, 19 ms of its
// 120 ms frames. What the results depend on is checked on the way back as after any other draw (Common::ResolveGroup:
// the values, the image changes since, each Held candidate's repeat), so they can wait out other programs' draws.
// That game's two programs hold 11 groups each, and with fewer places than that they evict each other in turn.
constexpr size_t MaxParkedGroups = 32;

// The slot's group waits among the parked ones when its results are reusable. The slot's own results are then invalid.
void ParkGroup(PreparedBindings& prepared, size_t slot) {
	auto& group = prepared.groups[slot];
	if (!group.reuse.valid) return;
	auto& parked = prepared.parked;
	const auto free = std::ranges::find_if(parked, [](const auto& place) { return !place.group.reuse.valid; });
	PreparedBindings::ParkedGroup* place = nullptr;
	if (free != parked.end()) {
		place = &*free;
	} else if (parked.size() < MaxParkedGroups) {
		place = &parked.emplace_back();
	} else {
		place = &parked[prepared.parked_next];
		prepared.parked_next = (prepared.parked_next + 1u) % MaxParkedGroups;
	}
	std::swap(place->members, prepared.image_groups[slot]);
	std::swap(place->group, group);
	group.reuse.valid = false;
}

// The slot, whose results are invalid, takes the parked results of this key and size if there are any.
void UnparkGroup(PreparedBindings& prepared, size_t slot, const ShaderRecompiler::IR::ImageResource& key, size_t size) {
	for (auto& place: prepared.parked) {
		if (place.group.reuse.valid && place.members.size() == size && place.group.reuse.key == key) {
			std::swap(place.members, prepared.image_groups[slot]);
			std::swap(place.group, prepared.groups[slot]);
			return;
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
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.program_constants = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.image_fills.resize(program.info.images.size());
	// A group can hold thousands of candidates: its vector keeps its storage from draw to draw, and its bindings while
	// they can be reused (GroupReuse). Every other entry is emptied.
	if (prepared.image_groups.size() < program.info.images.size()) prepared.image_groups.resize(program.info.images.size());
	prepared.groups.resize(prepared.image_groups.size());
	for (size_t i = 0; i < prepared.image_groups.size(); ++i) {
		if (i < program.info.images.size() && !program.info.images[i].heap &&
		    program.info.images[i].indirect_capacity != 0u) continue;
		ParkGroup(prepared, i);
		prepared.image_groups[i].clear();
		auto& group       = prepared.groups[i];
		group.reuse.valid = false;
		group.bound.clear();
		group.infos_whole = true;
	}
	prepared.samplers.clear();
	prepared.shader_data.clear();
	auto& texture_cache = m_context.GetTextureCache();
	uint64_t target_bytes_clock = UINT64_MAX;
	// Marvel's Wolverine's effect programs: what their store can reach, from guest memory (wolverineStores.h).
	WolverineStores::Reach store_reach;
	const bool             store_reach_known =
	    WolverineStores::IsStoreProgram(program.shader_hash) &&
	    WolverineStores::FindReach(program.shader_hash, std::span<const uint32_t>(snapshot.user_data), program.user_data_base,
	                               [&](uint64_t address, void* data, uint64_t bytes) {
		                               // Waits for the GPU, and the bytes are then what the GPU left. At the Trask stand it came
		                               // about 280 times in a run, about 55 ms in all, the longest 5.4 ms.
		                               auto& buffers = m_context.GetBufferCache();
		                               if (buffers.HasGpuDirtyBytes(address, bytes)) buffers.ReadMemory(address, bytes);
		                               return Libs::LibKernel::Memory::TryReadBufferBacking(address, data, bytes);
	                               },
	                               store_reach);
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (program.info.images[i].heap) {
			prepared.images[i]      = {};
			prepared.image_fills[i] = 0;
			continue;
		}
		if (program.info.images[i].indirect_capacity != 0u) {
			// The array's elements are the group's candidates; the image has no descriptor of its own.
			prepared.images[i]      = {};
			prepared.image_fills[i] = 0;
			// Resolving a storage group's thousands of candidates made most of Marvel's Wolverine's fight frames, and
			// the group was the same as that slot's previous one 95 % of the time. A candidate's null holds for good
			// when its descriptor alone makes it, and while no image is made or freed over its bytes when it lacks an
			// image (TextureCache::ImageChangesSince); a bound candidate whose plain lookup repeats and that no target
			// of the draw overlaps keeps its binding, with the lookup's side effects done again (RepeatLookup); the
			// others resolve again, in rank order (Common::ResolveGroup). A group that is only read resolves textures,
			// which can make images: it is resolved whole every time.
			const auto& resource = program.info.images[i];
			const auto& all_values = snapshot.image_groups[i];
			auto&       group    = prepared.groups[i];
			auto&       members  = prepared.image_groups[i];
			if (!group.reuse.valid || !(group.reuse.key == resource) || members.size() != all_values.size()) {
				ParkGroup(prepared, i);
				UnparkGroup(prepared, i, resource, all_values.size());
			}
			group.lookups.resize(all_values.size());
			// A candidate out of the store's reach can't be written: it resolves as a null, which holds until it is in reach.
			group.narrowed_now = store_reach_known && resource.written;
			if (group.narrowed_now) WolverineStores::Narrow(std::span(all_values), store_reach, group.narrowed, group.narrowed_kept);
			const auto& values = group.narrowed_now ? group.narrowed : all_values;
			// Marvel's Wolverine's Trask outpost repeats about 54,000 lookups a frame in its effect passes, about 25 draws
			// of 11-22 groups each, nearly all of them what the group's last draw found. A repeat reads the texture cache's
			// page table for the draw's targets and takes its lock three times. For bytes no target covers there is nothing
			// to find; and while the state a walk repeated all its lookups under stands, they hold, and their touches have
			// nothing left to do. A color image's clear check reads guest memory, so it stays on every draw. Whatever else
			// a draw does with a candidate follows as before: BindImage here, RebindImages and CommitBindings after.
			const auto state          = resource.written ? texture_cache.GetLookupState() : TextureCache::LookupState {};
			bool       lookups_hold   = resource.written && !state.grows && group.lookups_held == state;
			bool       lookups_failed = false;
			// Anything that makes, frees or links an image moves the clock: a stencil plane, or a merge, can add to the bytes.
			const auto read_targets = [&](uint64_t clock) {
				if (clock == target_bytes_clock) return;
				target_bytes_clock = clock;
				m_target_bytes.clear();
				texture_cache.AppendTargetBytes(m_target_images, m_target_bytes);
			};
			const auto after_change = [&] {
				const auto now = texture_cache.GetLookupState();
				if (now != state) lookups_hold = false;
				read_targets(now.clock);
			};
			if (resource.written) read_targets(state.clock);
			group.resolved.clear();
			const bool reused = Common::ResolveGroup(
			    std::span(values), resource, resource.written, members, group.reuse, ImageChanges {texture_cache},
			    [&](size_t rank, Common::GroupResult& result) {
				    TextureMemoValue made;
				    auto binding = resource.written ? ResolveTextureFully(resource, values[rank], true, &result, &made)
				                                    : ResolveTexture(resource, values[rank], true, &result);
				    if (binding.image_id && made.cacheable && made.mapped_size == 0u &&
				        result.outcome == Common::GroupOutcome::Recheck && result.end > result.address) {
					    result.outcome       = Common::GroupOutcome::Held;
					    group.lookups[rank] = {made.lookup, made.stamp, made.metadata_base_layer, result.address,
					                           result.end - result.address};
				    }
				    BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
				    if (resource.written) after_change();
				    group.resolved.push_back(static_cast<uint32_t>(rank));
				    if (rank >= members.size() || members[rank].image_id != binding.image_id) group.settled_valid = false;
				    return binding;
			    },
			    [&](size_t rank, TextureBinding& binding) {
				    // In the order the full resolve checks: the draw's targets, then the lookup.
				    const auto& lookup = group.lookups[rank];
				    const auto  end    = lookup.address + lookup.size;
				    if (std::ranges::any_of(m_target_bytes, [&](const auto& bytes) {
					        return lookup.address < bytes.second && bytes.first < end;
				        }) &&
				        texture_cache.HasTargetIn(lookup.address, lookup.size)) {
					    return false;
				    }
				    const bool clear_check = TextureCache::HasColorClearCheck(binding.desc);
				    if (!lookups_hold) {
					    if (!texture_cache.RepeatLookup(lookup.lookup, lookup.stamp, binding.desc,
					                                    lookup.metadata_base_layer)) {
						    lookups_failed = true;
						    return false;
					    }
				    } else if (clear_check) {
					    texture_cache.RepeatColorClear(lookup.lookup, binding.desc, lookup.metadata_base_layer);
				    }
				    if (clear_check) after_change();
				    BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
				    return true;
			    });
			// Every Held candidate now has its lookup done under this state: repeated or just resolved.
			group.lookups_held = {};
			if (reused && group.reuse.valid && resource.written && !state.grows && !lookups_failed &&
			    texture_cache.GetLookupState() == state) {
				group.lookups_held = state;
			}
			if (!reused || !group.reuse.valid) group.settled_valid = false;
			// The bound ranks, which RebindImages and CommitBindings visit; the others are nulls. Only visited ranks
			// can have changed since the last commit.
			group.bound.clear();
			if (!reused) {
				group.infos_whole = true;
				for (uint32_t rank = 0; rank < members.size(); ++rank) {
					if (members[rank].image_id) group.bound.push_back(rank);
				}
			} else {
				for (const auto rank: group.reuse.touched) {
					if (members[rank].image_id) group.bound.push_back(rank);
				}
				if (!group.infos_whole) group.dirty.insert(group.dirty.end(), group.reuse.touched.begin(), group.reuse.touched.end());
			}
			continue;
		}
		// A slot of a shader mostly sees the same T# draw after draw: it goes to its memo entry without hashing the key,
		// and while the prepared binding is a copy of that entry it isn't copied again. What the memo's hit does besides
		// finding the entry is the same (the mapping's check, the repeated lookup, the bind), and the binding's parts a
		// draw changes are put back, so it is what ResolveTexture returns. Programs and their resources live as long as
		// the pipeline cache, so the address names the slot.
		const auto& resource = program.info.images[i];
		auto&       slot     = (*m_image_slots)[(reinterpret_cast<uintptr_t>(&resource) * 0x9E3779B97F4A7C15ull) >> 51u];
		auto&       fill     = prepared.image_fills[i];
		if (slot.resource == &resource && slot.value == snapshot.images[i] && slot.entry->valid && slot.entry->value.fill == slot.fill) {
			const auto& memo = slot.entry->value;
			const auto& desc = memo.binding.desc;
			if ((memo.mapped_size == 0 || m_context.IsMapped(desc.info.data.address, memo.mapped_size)) &&
			    (!memo.lookup || texture_cache.RepeatLookup(memo.lookup, memo.stamp, desc, memo.metadata_base_layer))) {
				auto& kept = prepared.images[i];
				if (fill != slot.fill) {
					kept.image_id = memo.binding.image_id;
					kept.desc     = desc;
					fill          = slot.fill;
				}
				kept.image_view = memo.binding.image_view;
				kept.layout     = memo.binding.layout;
				kept.mip_views.clear();
				BindImage(kept.image_id, kept.desc.type == TextureCache::BindingType::Storage);
				continue;
			}
		}
		auto binding = ResolveTexture(resource, snapshot.images[i]);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
		fill = 0;
		if (m_resolved_entry != nullptr) {
			slot = {&resource, snapshot.images[i], m_resolved_entry, m_resolved_entry->value.fill};
			fill = slot.fill;
		}
	}
	prepared.heap_images.clear();
	if (auto* heap = m_context.GetGraphics().guest_image_heap; heap != nullptr && snapshot.image_heap.base != 0u) {
		namespace IR = ShaderRecompiler::IR;
		heap->Watch(snapshot.image_heap.base, snapshot.image_heap.size, m_context);
		// The game's heap entries that changed, and slots whose image changed layout or texels.
		const auto work  = heap->TakeWork(m_target_images);
		uint32_t   empty = 0;
		std::unordered_map<uint64_t, uint32_t> made; // address -> the slot that made an image there
		for (const auto& change: work) {
			IR::DescriptorValue descriptor {.dwords = change.descriptor, .dword_count = 8u};
			auto                info = IR::HeapImageInfo(descriptor, false);
			if (info == 0u) {
				heap->Clear(change.slot);
				++empty;
				continue;
			}
			// Post-processing and the final composite read render targets through the heap. Such a
			// slot, and one whose image went, gets a view of an image that exists, never one of its
			// own: the render targets of every dynamic-resolution step share one memory, each with an
			// entry (Wolverine Engine.md 5.2), and images made for them took 2-3 GB. Before any of
			// them is drawn, the first entry's image stands for the memory: each step's own took
			// 3.9 GB of the balcony title's 8.6 GB of heap images.
			const auto entry    = DecodeNativeDescriptor<ShaderTextureResource>(descriptor);
			const auto address  = entry.Base40();
			// The guest's mip tail stays whole: as a texture of their own, levels from inside it lie
			// elsewhere in its tail block. The image starts at the tail's first level at the latest;
			// the view's min LOD still keeps sampling at the loaded level.
			if (const auto loaded = (info >> IR::ImageHeapLevelShift) & 0xfu; loaded != 0u) {
				const auto tail = FirstTailLevel(entry);
				if (loaded > tail) info = (info & ~(0xfu << IR::ImageHeapLevelShift)) | (tail << IR::ImageHeapLevelShift);
			}
			const auto first    = (info >> IR::ImageHeapLevelShift) & 0xfu;
			const bool held     = first != 0u && m_context.GetTextureCache().HoldsLevels(
			                                     entry.Format(), entry.TileMode(), entry.Width5() + 1u,
			                                     entry.Height5() + 1u, entry.MetaCompress(), first);
			const auto maker    = made.find(address);
			const bool existing = heap->Waiting(change.slot) ||
			                      m_context.GetTextureCache().HasGpuImageAt(address) ||
			                      (maker != made.end() && maker->second != change.slot) ||
			                      m_context.GetTextureCache().HasHeapImageAt(address, heap->Image(change.slot));
			bool       waits    = false;
			bool       create   = !existing;
			for (const auto dimension: {ShaderRecompiler::Decoder::ImageDimension::Dim2D,
			                            ShaderRecompiler::Decoder::ImageDimension::Dim2DArray,
			                            ShaderRecompiler::Decoder::ImageDimension::Dim3D,
			                            ShaderRecompiler::Decoder::ImageDimension::Dim1D}) {
				if ((info & IR::ImageHeapFilled(dimension)) == 0u) continue;
				PreparedBindings::HeapImage image;
				image.descriptor              = descriptor;
				image.slot                    = change.slot;
				image.info                    = info;
				image.resource.resource_class = IR::ImageResourceClass::Sampled;
				image.resource.numeric_class  = static_cast<Prospero::TextureNumericClass>(
				    (info & 3u) + static_cast<uint32_t>(Prospero::TextureNumericClass::Float));
				image.resource.dimension = dimension;
				image.resource.cube      = (info & IR::ImageHeapCube) != 0u;
				image.resource.read      = true;
				DropLevels(image.descriptor, first, held);
				image.texture = ResolveTexture(image.resource, image.descriptor, create);
				// A streamed texture (a mip chain) over memory no shader writes is made from it like any
				// texture: another slot's version with fewer levels, or the loss of the slot's image, left
				// it on the null image until the game wrote its entry again (Logan's forearms went black).
				if (const auto levels = DecodeNativeDescriptor<ShaderTextureResource>(image.descriptor);
				    !image.texture.image_id && existing && levels.LastLevel() > levels.BaseLevel() &&
				    !m_context.GetTextureCache().HasGpuImageAt(address)) {
					image.texture = ResolveTexture(image.resource, image.descriptor, true);
				}
				// A slot waits again when its image goes (GuestImageHeap::Forget: an unmap, the collector, a later
				// resolve over the same memory), and was then resolved from existing images only. Where no image of
				// any kind lies, nothing else makes one, and the slot stayed empty until the game wrote it again:
				// Marvel's Wolverine's fire lights (8x8 textures) went black after a checkpoint reload. It is made
				// from memory, as its first lookup was. Over memory another slot's or the GPU's image covers (the
				// dynamic-resolution steps, render targets) it still waits. KYTY_HEAP_WAIT_ANY: off switch.
				static const bool wait_any = std::getenv("KYTY_HEAP_WAIT_ANY") != nullptr;
				// shortcut: the mapped check covers the first byte only; the game maps a texture whole.
				if (!image.texture.image_id && !create && !wait_any && m_context.IsMapped(address, 1u) &&
				    !m_context.GetTextureCache().HasAnyImageAt(address)) {
					create        = true; // the slot's other dimensions too
					image.texture = ResolveTexture(image.resource, image.descriptor, true);
				}
				if (!image.texture.image_id) {
					waits = true;
					continue;
				}
				// A held texture's view starts at its resident levels until a grow is bound: the slot
				// says the same level, so explicit LODs and fetches count from it.
				if (m_context.GetTextureCache().GetImage(image.texture.image_id).held) {
					image.info = (image.info & ~(0xfu << IR::ImageHeapLevelShift)) |
					             (image.texture.desc.view_info.base_level << IR::ImageHeapLevelShift);
				}
				BindImage(image.texture.image_id, false);
				prepared.heap_images.push_back(std::move(image));
			}
			if (waits) {
				heap->Wait(change.slot, address);
				++empty;
			} else if (create) {
				made.emplace(address, change.slot);
			}
		}
		// Only when the game wrote entries or a slot retries: stale slots alone come many times a frame, on the draw path.
		if (work.size() != heap->last_stale) {
			LOGF("Image heap: %zu entries resolved (%u stale, %u retried), %u empty\n",
			     work.size(), heap->last_stale, heap->last_retried, empty);
		}
	}
	if (auto* heap = m_context.GetGraphics().guest_image_heap; heap != nullptr && snapshot.sampler_heap.base != 0u) {
		const auto sampler_changes = heap->SamplerChanges(snapshot.sampler_heap.base, snapshot.sampler_heap.size);
		for (const auto& [slot, record]: sampler_changes) {
			ShaderRecompiler::IR::DescriptorValue value {.dword_count = 4u};
			std::copy(record.begin(), record.end(), value.dwords.begin());
			auto descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
			// Made as the shader's own samplers are (NativeSampler): heap samples never compare.
			descriptor.fields[0] &= ~(0x7u << 12u);
			const auto float_sampler = m_context.GetSamplerCache().GetSampler(descriptor, false);
			descriptor.SetPointFiltering();
			heap->ShowSampler(slot, float_sampler, m_context.GetSamplerCache().GetSampler(descriptor, true));
		}
	}
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto& prepared = *stage;
		EXIT_IF((prepared.runtime == nullptr || !*prepared.runtime) && !KYTY_UPSTREAM_PATH("e827c92a2"));
		const auto& program  = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout = program.bindings;
		prepared.buffer_sources.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
			    !metadata.scalar) {
				// OOB_SELECT=2 only tests NumRecords != 0; a zero-stride fetch
				// still needs its full payload when NumRecords is smaller than it.
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (address == 0 || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					// Sentinel write ranges end before tables captured by any bound stage.
					for (const auto* reader: stages) {
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				}
				//  PPSA15246
				const auto& graphics = m_context.GetGraphics();
				const auto  limit =
				    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				    (graphics.StorageMinAlignment() - 1);
				size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
			}
			// Finite GPU descriptors can span sparse PRT holes; clamp only oversized/sentinel ranges. An oversized finite one
			// also spans reserved space and stops only where the address space is free (Marvel's Wolverine's geometry pool with
			// the game's lazy commit: one 2.7 GB view over 2 MiB pages mapped as blocks land; unmapped pages read as zero).
			if (size >= (uint64_t {1} << 30u) || descriptor.NumRecords() == UINT32_MAX) {
				size = Libs::LibKernel::Memory::ClampRangeSize(address, size, descriptor.NumRecords() != UINT32_MAX);
			}
			source = {address, size, cache.FindBuffer(address, size)};
		}
	}
}

// A later record's draw index, in the stage's shader data, and uploaded if it goes that way. The rest of the shader data is
// what the record before it had, as nothing has reached the caches. A mesh stage's user data starts at SGPR 8. False when the
// stage doesn't read that slot.
bool RenderExecutor::UploadDrawIndex(RenderContext& context, PreparedBindings& stage, uint32_t slot, uint32_t draw_index,
                                     uint32_t& dword) {
	const auto& program   = *stage.runtime->program;
	const auto& registers = program.info.user_data_registers;
	for (dword = 0; dword < registers.size(); ++dword) {
		if (registers[dword] - program.user_data_base != slot + 8u) continue;
		stage.shader_data[dword] = draw_index;
		if (!program.bindings.UsesPushData()) stage.shader_data_buffer = NativeUpload(context, stage.shader_data);
		return true;
	}
	return false;
}

// By the way the full draw took it: the push data's dword, or the same descriptor writes of set 0 with the new upload.
void RenderExecutor::CommitDrawIndex(CommandBuffer& buffer, uint32_t dword, uint32_t draw_index) {
	const auto& later     = m_later_records;
	const auto& bindings  = later.stage->runtime->program->bindings;
	const auto  vk_buffer = buffer.Recorder();
	if (bindings.UsesPushData()) {
		vk_buffer.pushConstants(later.layout, m_commit_push_stages, (bindings.push_data_start_dword + dword) * sizeof(uint32_t),
		                        sizeof(uint32_t), &draw_index);
		return;
	}
	const auto writes = std::span(m_descriptor_writes).first(m_commit_split);
	if (later.push_descriptors) {
		vk_buffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, later.layout, 0, static_cast<uint32_t>(writes.size()),
		                               writes.data());
		return;
	}
	const auto set = m_context.GetDescriptorHeap().Commit(later.set_layout);
	for (auto& write: writes) {
		write.dstSet = set;
	}
	// The fresh set's update goes with its bind (design rule 5), as in CommitBindings.
	vk_buffer.UpdateDescriptorSets(m_context.GetGraphics().device, static_cast<uint32_t>(writes.size()), writes.data());
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, later.layout, 0, 1, &set, 0, nullptr);
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF((prepared.runtime == nullptr || !*prepared.runtime) && !KYTY_UPSTREAM_PATH("e827c92a2"));
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF((prepared.buffer_sources.size() != layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]) && !KYTY_UPSTREAM_PATH("e827c92a2"));

	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	EXIT_IF((prepared.shader_data.size() != layout.ShaderDataDwords()) && !KYTY_UPSTREAM_PATH("e827c92a2"));
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		uint64_t track         = 0;
		Buffer*  bound         = nullptr;
		const auto buffer = static_cast<size_t>(&resource - program.info.buffers.data());
		prepared.buffers[index] = NativeStorageBuffer(
		    m_context, prepared.buffer_sources[index], resource, buffer_offset,
		    buffer < snapshot.buffer_writes.size() ? snapshot.buffer_writes[buffer]
		                                           : std::pair<uint64_t, uint64_t> {0, UINT64_MAX},
		    buffer < snapshot.buffer_write_ranges.size() ? std::span(snapshot.buffer_write_ranges[buffer])
		                                                 : std::span<const std::pair<uint64_t, uint64_t>> {},
		    layout.write_tracking ? &track : nullptr, layout.BufferTable() ? &bound : nullptr);
		pack_memory_offset(index, buffer_offset);
		if (track != 0) {
			prepared.shader_data[layout.TrackDword() + 2u * index]      = static_cast<uint32_t>(track);
			prepared.shader_data[layout.TrackDword() + 2u * index + 1u] = static_cast<uint32_t>(track >> 32u);
		}
		// The shader's bounds: what OpArrayLength would give (NativeStorageBuffer keeps it in 32 bits),
		// and 0 for an empty guest buffer, bound to the shared null buffer, which nothing may write.
		const auto& source = prepared.buffer_sources[index];
		prepared.shader_data[layout.BufferLengthDword() + index] =
		    source.address == 0 || source.size == 0 ? 0u
		                                            : static_cast<uint32_t>(prepared.buffers[index].range);
		if (layout.BufferTable()) {
			// The base address the descriptor would name; 0 for an empty buffer (length 0).
			uint64_t base = 0;
			if (source.address != 0 && source.size != 0) {
				// Can't happen: a non-empty source gets a game buffer from the cache (ObtainBuffer; FindBuffer gives the null id
				// only for address 0), and game buffers are always created with eShaderDeviceAddress. Only the null buffer has
				// none, and null sources take base 0 above, which the shader reads as an empty buffer. Reaching this is a bug
				// elsewhere; a silent 0 would hide it as a gap on screen.
				if (bound == nullptr) EXIT("buffer table: a binding without a device address\n");
				base = bound->BufferDeviceAddress() + prepared.buffers[index].offset;
			}
			prepared.shader_data[layout.TableDword() + 2u * index]      = static_cast<uint32_t>(base);
			prepared.shader_data[layout.TableDword() + 2u * index + 1u] = static_cast<uint32_t>(base >> 32u);
		}
	}
	if (layout.image_forms) prepared.shader_data[layout.ImageFormDword()] = snapshot.image_forms;
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ProgramConstants)] != 0) {
		const auto& buffer = m_context.GetBufferCache().GetProgramConstants(program.program_constants);
		prepared.program_constants = {buffer.Handle(), 0, buffer.Size()};
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

namespace {

// The ranks of a group over one image.
void AddImageRanks(PreparedBindings::Group& group, uint32_t image) {
	const auto from = std::ranges::lower_bound(group.image_ranks, std::pair {image, uint32_t {0}});
	for (auto at = from; at != group.image_ranks.end() && at->first == image; ++at) group.visits.push_back(at->second);
}

// After a walk of all a group's bound ranks: which image each is over, and the state that holds under. A transit can go
// on leaving ranks out only while they are the same images: an id can come back as another image when the clock moves.
void SettleGroup(PreparedBindings::Group& group, const std::vector<TextureBinding>& members, bool can_settle, bool same_clock,
                 const TextureCache::LookupState& state) {
	group.settled_valid = can_settle && !state.grows;
	group.settled       = state;
	if (!group.settled_valid || !same_clock) group.transit_valid = false;
	// Nearly always the same images as last time, which takes no sort to see.
	size_t bound = 0;
	for (const auto rank: group.bound) bound += members[rank].image_id ? 1u : 0u;
	if (bound == group.image_ranks.size() && std::ranges::all_of(group.image_ranks, [&](const auto& entry) {
		    return members[entry.second].image_id.index == entry.first;
	    })) {
		return;
	}
	group.transit_valid = false;
	group.image_ranks.clear();
	group.always.clear();
	for (const auto rank: group.bound) {
		const auto& member = members[rank];
		if (!member.image_id) continue;
		group.image_ranks.emplace_back(member.image_id.index, rank);
		if (member.desc.type != TextureCache::BindingType::Storage) group.always.push_back(rank);
	}
	std::ranges::sort(group.image_ranks);
}

} // namespace

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF((prepared.runtime == nullptr || !*prepared.runtime) && !KYTY_UPSTREAM_PATH("e827c92a2"));
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF((images.size() != program.info.images.size()) && !KYTY_UPSTREAM_PATH("e827c92a2"));
	// Groups come from PrepareBindings only; the vector never shrinks.
	if (prepared.image_groups.size() < images.size()) prepared.image_groups.resize(images.size());
	prepared.image_fills.resize(images.size()); // likewise: bindings can come from elsewhere
	auto& texture_cache = m_context.GetTextureCache();
	bool in_group            = false;
	bool group_image_changed = false;
	// Acquire each view before a later overlapping descriptor can replace its image.
	const auto rebind = [&](TextureBinding& binding, const ShaderRecompiler::IR::ImageResource& resource,
	                        const ShaderRecompiler::IR::DescriptorValue& value, uint64_t* fill = nullptr) {
		if (!binding.image_id) return;
		const auto first_image = binding.image_id;
		const auto old_image = texture_cache.m_slot_images.try_get(binding.image_id);
		if (old_image == nullptr || !old_image->registered ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			binding = ResolveTexture(resource, value);
			if (fill != nullptr) *fill = 0;
			BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
			if (!binding.image_id) { // a storage candidate bound as null (ResolveTextureFully), as the heap path below
				if (in_group) group_image_changed = true;
				binding.image_view = nullptr;
				binding.mip_views.clear();
				return;
			}
		}
		binding.mip_views.clear();
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		if (in_group) {
			if (binding.image_id != first_image) group_image_changed = true;
			image.EnrollGroupNotices(texture_cache.m_group_notices, binding.image_id.index);
		}
	};
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto& resource = program.info.images[i];
		if (resource.indirect_capacity == 0u) {
			rebind(images[i], resource, snapshot.images[i], &prepared.image_fills[i]);
			continue;
		}
		auto& group = prepared.image_groups[i];
		EXIT_IF(group.size() != snapshot.image_groups[i].size() || i >= prepared.groups.size());
		auto& walk = prepared.groups[i];
		const auto& values = walk.narrowed_now ? walk.narrowed : snapshot.image_groups[i];
		in_group            = resource.written;
		group_image_changed = false;
		const auto notices_end = texture_cache.m_group_notices.End();
		const auto state       = resource.written ? texture_cache.GetLookupState() : TextureCache::LookupState {};
		bool       visited     = false;
		if (resource.written && walk.settled_valid && walk.settled == state && !state.grows) {
			// Nothing was made or freed and this tick's touches are done, so only the ranks that resolved in this draw and
			// those over an image with a change have anything for FindTexture to do.
			walk.visits = walk.resolved;
			visited     = texture_cache.m_group_notices.Since(walk.rebind_notices, notices_end, [&](uint32_t image, GroupNotices::Kind kind) {
				if (kind == GroupNotices::Kind::Contents) AddImageRanks(walk, image);
			});
			if (visited) {
				std::ranges::sort(walk.visits);
				const auto [first, last] = std::ranges::unique(walk.visits);
				walk.visits.erase(first, last);
				for (const auto rank: walk.visits) rebind(group[rank], resource, values[rank]);
				if (group_image_changed || texture_cache.GetLookupState() != state) walk.settled_valid = false;
			}
		}
		if (!visited) {
			for (const auto rank: walk.bound) rebind(group[rank], resource, values[rank]);
			SettleGroup(walk, group, resource.written && walk.reuse.valid, walk.settled.clock == state.clock, texture_cache.GetLookupState());
		}
		walk.rebind_notices = notices_end;
		in_group            = false;
	}
	for (auto& entry: prepared.heap_images) {
		auto&      binding   = entry.texture;
		const auto old_image = texture_cache.m_slot_images.try_get(binding.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			binding = ResolveTexture(entry.resource, entry.descriptor, false);
			if (!binding.image_id) {
				binding.image_view = nullptr;
				continue;
			}
			if (texture_cache.GetImage(binding.image_id).held) {
				namespace IR = ShaderRecompiler::IR;
				entry.info   = (entry.info & ~(0xfu << IR::ImageHeapLevelShift)) |
				               (binding.desc.view_info.base_level << IR::ImageHeapLevelShift);
			}
			BindImage(binding.image_id, false);
		}
		binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma        = false;
	bool uses_image_heap = false;
	FindBuffers(stages);
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->program->info.uses_dma;
		uses_image_heap |= stage->runtime->program->info.uses_image_heap;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	if (uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		auto& old_image = cache.m_slot_images[target.image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding = {};
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. The caller reserves the draw's buffers after image
	// identities are final and after its vertex, index and argument buffers; attachment layout transitions follow buffer
	// alias copies.
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	auto   vk_buffer        = buffer.Recorder();
	size_t image_descriptor_count = 0;
	size_t write_count      = 0;
	bool   uses_guest_heap  = false;
	auto*  image_heap       = m_context.GetGraphics().guest_image_heap;
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
		EXIT_IF((prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime) && !KYTY_UPSTREAM_PATH("e827c92a2"));
		const auto& program = *prepared->runtime->program;
		write_count += std::popcount(program.bindings.descriptor_mask);
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	if (pipeline.task_stage) {
		push_stages |= vk::ShaderStageFlagBits::eTaskEXT;
	}
	// The written images' and buffers' ranges, gathered once: a draw binds hundreds of images and writes a
	// few, and walking all of them for every scalar read cost 4-6 ms a frame on Marvel's Wolverine's title scenes.
	// A storage-candidate group writes ~1,400 a stage there in its fight: the images' ranges are sorted by start, with
	// the furthest end up to each, so a read finds whether any overlaps it by one search.
	auto&                                      written_images = m_written_images;
	auto&                                      written_ends   = m_written_ends;
	std::vector<std::pair<uint64_t, uint64_t>> written_buffers;
	written_images.clear();
	bool                                       written_images_known = false;
	// An assertion that changes no draw; checking runs turn it on (scalarReadCheck.h).
	const bool check_scalar_reads = ScalarReadCheck::Enabled();
	for (const auto* reader: prepared_bindings) {
		if (!check_scalar_reads) break;
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		for (const auto* writer: prepared_bindings) {
			if (writer->runtime->program->has_address_writes) {
				EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
			}
		}
		if (!written_images_known) {
			written_images_known = true;
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0) written_images.emplace_back(written.address, written.size);
				}
			}
			std::sort(written_images.begin(), written_images.end());
			written_ends.resize(written_images.size());
			uint64_t furthest = 0;
			for (size_t at = 0; at < written_images.size(); ++at) {
				const auto [written_address, written_size] = written_images[at];
				if (written_address > UINT64_MAX - written_size) EXIT("invalid image overlap range\n");
				furthest         = std::max(furthest, written_address + written_size);
				written_ends[at] = furthest;
			}
			for (const auto* writer: prepared_bindings) {
				for (const auto& resource: writer->runtime->program->info.buffers) {
					if (!resource.written || resource.descriptor_index == UINT32_MAX) continue;
					const auto& written = writer->buffer_sources[resource.descriptor_index];
					// ADD_TID and swizzle move fixed-offset writes past write_extent.
					const bool linear = (resource.packed_stride & ((1u << 14u) | (1u << 20u))) == 0u;
					const auto written_size =
					    linear ? std::min<uint64_t>(written.size, resource.write_extent) : written.size;
					if (written_size != 0) written_buffers.emplace_back(written.address, written_size);
				}
			}
		}
		for (const auto [address, size]: reads) {
			if (!written_images.empty()) {
				if (size == 0 || address > UINT64_MAX - size) EXIT("invalid image overlap range\n");
				if (ScalarReadCheck::OverlapsWritten(written_images, written_ends, address, size)) {
					EXIT("scalar resource reads overlap an image or attachment write\n");
				}
			}
			for (const auto [written_address, written_size]: written_buffers) {
				if (ImageRangeOverlaps(address, size, written_address, written_size)) {
					// The reads were evaluated from the state before the dispatch, which is
					// what a shader that reads ahead of its own store sees (Wolverine CS
					// 0x85a58319a7a75e48 reads a branch input from a 256-byte buffer it
					// writes 16 bytes of).
					static std::atomic_flag warned;
					if (!warned.test_and_set()) {
						fprintf(stderr, "warning: scalar resource reads overlap a shader buffer write\n");
					}
				}
			}
		}
	}
	if (pipeline.uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, pipeline_bind_point, pipeline.pipeline_layout);
	}
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_runs.clear();
	m_descriptor_writes.clear();
	m_descriptor_writes.reserve(write_count);
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;
	size_t                                  fragment_first_write = SIZE_MAX; // the pixel shader's, for its own set

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (shader_stage == vk::ShaderStageFlagBits::eFragment) fragment_first_write = m_descriptor_writes.size();
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		// Storage images already in their state (written by an earlier draw and bound again) take one memory barrier
		// with the masks their own barriers would have had: a storage candidate group binds ~130 written images a draw
		// in Marvel's Wolverine's fight, and a barrier call each cost more than the rest of their binding.
		vk::MemoryBarrier2 repeated {};
		const auto transit = [&](TextureBinding& binding) {
			if (!binding.image_id) return;
			auto& image   = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.binding.is_target) {
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
				constexpr auto access = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
				if (image.InState(vk::ImageLayout::eGeneral, access, range)) {
					const auto stage = Image::TransitStage(access);
					repeated.srcStageMask |= stage;
					repeated.srcAccessMask |= access;
					repeated.dstStageMask |= stage;
					repeated.dstAccessMask |= access;
				} else {
					image.Transit(vk::ImageLayout::eGeneral, access, range, vk_buffer);
				}
				binding.layout = vk::ImageLayout::eGeneral; // backing.state: the last transition on any subresource
				return;
			} else {
				image.Transit(image.ReadLayout(), vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		};
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			transit(descriptors.images[i]);
			if (i < descriptors.image_groups.size() && !descriptors.image_groups[i].empty()) {
				auto&      walk        = descriptors.groups[i];
				auto&      members     = descriptors.image_groups[i];
				auto&      notices     = m_context.GetTextureCache().m_group_notices;
				const auto notices_end = notices.End();
				bool       visited     = false;
				if (walk.settled_valid && walk.transit_valid) {
					// A bound candidate is a storage image left in General with the masks of the last visit until it gets a
					// notice, and this draw's own targets and forced images take their branch of the transit whatever they were.
					walk.visits = walk.resolved;
					walk.visits.insert(walk.visits.end(), walk.always.begin(), walk.always.end());
					walk.visits.insert(walk.visits.end(), walk.held_targets.begin(), walk.held_targets.end());
					visited = notices.Since(walk.transit_notices, notices_end, [&](uint32_t image, GroupNotices::Kind kind) {
						if (kind == GroupNotices::Kind::State) AddImageRanks(walk, image);
					});
					if (visited) {
						for (const auto id: m_target_images) AddImageRanks(walk, id.index);
						for (const auto id: m_force_general_images) AddImageRanks(walk, id.index);
						std::ranges::sort(walk.visits);
						const auto [first, last] = std::ranges::unique(walk.visits);
						walk.visits.erase(first, last);
						size_t bound_visits = 0;
						for (const auto rank: walk.visits) {
							if (members[rank].image_id) ++bound_visits;
							transit(members[rank]);
						}
						if (bound_visits < walk.bound.size()) {
							constexpr auto access = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
							const auto     stage  = Image::TransitStage(access);
							repeated.srcStageMask |= stage;
							repeated.srcAccessMask |= access;
							repeated.dstStageMask |= stage;
							repeated.dstAccessMask |= access;
						}
					}
				}
				if (!visited) {
					for (const auto rank: walk.bound) transit(members[rank]);
					walk.transit_valid = walk.settled_valid;
				}
				walk.transit_notices = notices_end;
				// A notice isn't used up while its image is a target: the next draw that doesn't render to it moves it back.
				walk.visits.clear();
				for (const auto id: m_target_images) AddImageRanks(walk, id.index);
				walk.held_targets.swap(walk.visits);
			}
		}
		for (auto& entry: descriptors.heap_images) {
			if (!entry.texture.image_id) continue; // no image left at its rebind (RebindImages)
			// shortcut: a slot whose image this draw renders to is not shown now; give it the
			// attachment's layout when a game samples its own target through the heap.
			if (m_context.GetTextureCache().GetImage(entry.texture.image_id).binding.is_target) {
				entry.texture.image_view = nullptr;
				continue;
			}
			transit(entry.texture);
		}
		if (repeated.srcStageMask) {
			buffer.EndRendering();
			vk::DependencyInfo dependency {};
			dependency.memoryBarrierCount = 1;
			dependency.pMemoryBarriers    = &repeated;
			vk_buffer.pipelineBarrier2(dependency);
		}

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						EXIT_IF((descriptors.buffers.size() != write.descriptorCount) && !KYTY_UPSTREAM_PATH("e827c92a2"));
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
						}
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto& cache = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto& view = address_buffers[kind == BindingKind::FaultBuffer];
						view = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::ProgramConstants:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						} else if (kind == BindingKind::ProgramConstants) {
							view = &descriptors.program_constants;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							EXIT_IF((sampler == nullptr) && !KYTY_UPSTREAM_PATH("86139e36f"));
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture = descriptors.images[i];
			if (resource.heap) continue;
			EXIT_IF(texture.image_id &&
			        resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()) &&
			        !KYTY_UPSTREAM_PATH("e827c92a2"));
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			if (resource.guest_heap) {
				// What a heap read of an empty slot falls back to: slot 0 holds the null image.
				image_heap->Show(0u, ShaderRecompiler::IR::ImageBindingIndex(kind), texture.image_id,
				                 texture.image_view, texture.layout, 0u);
				uses_guest_heap = true;
				continue;
			}
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			if (resource.indirect_capacity != 0u) {
				// A group's array: its candidates by rank, from the group's own image infos, of which only the ranks
				// visited since the last commit can have changed; the rest of its capacity stays unwritten.
				EXIT_IF(i >= descriptors.image_groups.size());
				const auto& members = descriptors.image_groups[i];
				auto&       group   = descriptors.groups[i];
				EXIT_IF(members.size() > resource.indirect_capacity);
				bool       changed = group.infos.size() != members.size() || group.generation == 0u;
				const auto set     = [&](size_t rank) {
					const auto info = MakeImageInfo(members[rank]);
					if (!(info == group.infos[rank])) changed = true;
					group.infos[rank] = info;
				};
				if (group.infos_whole || group.infos.size() != members.size()) {
					group.infos.resize(members.size());
					for (size_t rank = 0; rank < members.size(); ++rank) set(rank);
				} else {
					for (const auto rank: group.dirty) set(rank);
				}
				if (changed) group.generation = g_group_generation.fetch_add(1, std::memory_order_relaxed) + 1u;
				group.dirty.clear();
				group.infos_whole = false;
				if (ShaderRecompiler::IR::GroupSetFor(program.stage) != ShaderRecompiler::IR::NoGroupSet) continue; // its own set, below
				if (!members.empty()) m_descriptor_runs.push_back({start, start + members.size(), group.infos.data()});
				if (members.size() < resource.indirect_capacity)
					m_descriptor_runs.push_back({start + members.size(), start + resource.indirect_capacity, nullptr});
				continue;
			}
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}

		for (const auto& entry: descriptors.heap_images) {
			if (entry.texture.image_view == nullptr) continue;
			auto& image = m_context.GetTextureCache().GetImage(entry.texture.image_id);
			// A later resolve of this draw can free an image an earlier entry got (two entries over
			// one memory): its view goes with it, and GuestImageHeap::Forget has already run for it.
			if (!image.registered && !image.info.data.Empty()) {
				image_heap->Retry(entry.slot);
				continue;
			}
			// Its layout and texel changes reach the heap (Image::GetBarriers, MarkCpuDirty).
			image.heap_id = entry.texture.image_id;
			image_heap->Show(entry.slot,
			                 ShaderRecompiler::IR::ImageBindingIndex(
			                     *ShaderRecompiler::IR::DescriptorBindingForImage(entry.resource)),
			                 entry.texture.image_id, entry.texture.image_view, entry.texture.layout,
			                 entry.info);
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF((prepared->shader_data.size() != shader_data_dwords) && !KYTY_UPSTREAM_PATH("e827c92a2"));
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

	// The pixel shader's writes go to its own set where the pipeline has one (GraphicContext::SplitsFragmentSet).
	auto split = pipeline.fragment_set_layout != nullptr ? std::min(fragment_first_write, m_descriptor_writes.size())
	                                                     : m_descriptor_writes.size();
	if (!m_descriptor_runs.empty()) {
		// A write over a group becomes writes of the runs around it, of the group's array from its own image infos,
		// and none over its tail (image arrays are partially bound; a push-descriptor set's are not).
		std::sort(m_descriptor_runs.begin(), m_descriptor_runs.end());
		std::vector<vk::WriteDescriptorSet> runs;
		runs.reserve(m_descriptor_writes.size() + m_descriptor_runs.size() * 2u);
		size_t runs_split = SIZE_MAX;
		for (size_t index = 0; index < m_descriptor_writes.size(); ++index) {
			if (index == split) runs_split = runs.size();
			const auto& write = m_descriptor_writes[index];
			if (write.pImageInfo == nullptr) {
				runs.push_back(write);
				continue;
			}
			const size_t begin = write.pImageInfo - m_descriptor_images.data();
			const size_t end   = begin + write.descriptorCount;
			const auto   add   = [&](size_t from, size_t to, const vk::DescriptorImageInfo* source) {
				auto run            = write;
				run.dstArrayElement = write.dstArrayElement + static_cast<uint32_t>(from - begin);
				run.pImageInfo      = source;
				run.descriptorCount = static_cast<uint32_t>(to - from);
				runs.push_back(run);
			};
			size_t next = begin;
			for (const auto& [run_begin, run_end, source]: m_descriptor_runs) {
				if (run_end <= next || run_begin >= end) continue;
				if (run_begin > next) add(next, run_begin, m_descriptor_images.data() + next);
				const auto from = std::max(next, run_begin);
				const auto to   = std::min(end, run_end);
				if (source != nullptr) {
					add(from, to, source + (from - run_begin));
				} else {
					EXIT_IF(index < split ? pipeline.uses_push_descriptors : pipeline.fragment_uses_push_descriptors);
				}
				next = to;
			}
			if (next < end) add(next, end, m_descriptor_images.data() + next);
		}
		split = runs_split == SIZE_MAX ? runs.size() : runs_split;
		m_descriptor_writes.swap(runs);
	}
	const auto commit = [&](size_t first, size_t last, uint32_t set_index, vk::DescriptorSetLayout layout,
	                        bool uses_push_descriptors) {
		if (first == last) return;
		EXIT_IF(layout == nullptr);
		const auto writes = std::span(m_descriptor_writes).subspan(first, last - first);
		if (uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, set_index,
			                               static_cast<uint32_t>(writes.size()), writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(layout);
			for (auto& write: writes) {
				write.dstSet = set;
			}
			// The fresh set's update goes with its bind (design rule 5).
			vk_buffer.UpdateDescriptorSets(m_context.GetGraphics().device, static_cast<uint32_t>(writes.size()),
			                               writes.data());
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, set_index, 1, &set, 0,
			                             nullptr);
		}
	};
	m_commit_split       = split;
	m_commit_push_stages = push_stages;
	commit(0, split, 0, pipeline.descriptor_set_layout, pipeline.uses_push_descriptors);
	if (uses_guest_heap) {
		// Pipeline layouts differ in set 0, so the heap is bound again for each. Before set 2: binding set 1 under
		// a layout the last binding of set 1 is not compatible with disturbs every set above it.
		const auto set = image_heap->Set();
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout,
		                             ShaderRecompiler::IR::ImageHeapSet, 1, &set, 0, nullptr);
	}
	commit(split, m_descriptor_writes.size(), ShaderRecompiler::IR::FragmentSet, pipeline.fragment_set_layout,
	       pipeline.fragment_uses_push_descriptors);
	// The image groups' own sets, last (a set bound under a layout disturbs none below it). A stage's kept set is bound
	// again while its groups' infos are what it was written from and no image whose views a kept set holds was
	// destroyed; otherwise a set is written whole, its nulls too, since a shader may index them.
	for (const auto* prepared: prepared_bindings) {
		const auto& program   = *prepared->runtime->program;
		const auto  set_index = ShaderRecompiler::IR::GroupSetFor(program.stage);
		if (set_index == ShaderRecompiler::IR::NoGroupSet ||
		    std::ranges::none_of(program.bindings.group_counts, [](uint32_t count) { return count != 0u; })) {
			continue;
		}
		const auto layout = pipeline.group_set_layouts[set_index - ShaderRecompiler::IR::GroupSet];
		EXIT_IF(layout == nullptr);
		m_group_generations.clear();
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			if (program.info.images[i].indirect_capacity != 0u) m_group_generations.push_back(prepared->groups[i].generation);
		}
		const auto epoch = Image::group_set_epoch.load(std::memory_order_relaxed);
		auto&      kept  = m_kept_group_sets[&program];
		auto       set   = kept.set;
		if (set == nullptr || kept.epoch != epoch || kept.generations != m_group_generations) {
			auto& heap = m_context.GetDescriptorHeap();
			if (kept.set != nullptr) heap.ReleaseKept(kept.set);
			kept.set         = heap.AllocateKept(layout);
			kept.epoch       = epoch;
			kept.generations = m_group_generations;
			set              = kept.set != nullptr ? kept.set : heap.Commit(layout); // the pool full: a per-frame set
			m_group_writes.clear();
			for (uint32_t i = 0; i < program.info.images.size(); ++i) {
				const auto& resource = program.info.images[i];
				const auto& members  = prepared->image_groups[i];
				if (resource.indirect_capacity == 0u || members.empty()) continue;
				const auto& infos = prepared->groups[i].infos;
				EXIT_IF(infos.size() != members.size());
				const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
				vk::WriteDescriptorSet write {};
				write.dstSet          = set;
				write.dstBinding      = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
				write.dstArrayElement = resource.descriptor_index;
				write.descriptorCount = static_cast<uint32_t>(members.size());
				write.descriptorType  = NativeDescriptorType(kind);
				write.pImageInfo      = infos.data();
				m_group_writes.push_back(write);
				if (kept.set == nullptr) continue;
				for (const auto& member: members) {
					if (member.image_id) m_context.GetTextureCache().GetImage(member.image_id).in_group_set = true;
				}
			}
			m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(m_group_writes.size()),
			                                                    m_group_writes.data(), 0, nullptr);
		}
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, set_index, 1, &set, 0, nullptr);
	}
}

} // namespace Libs::Graphics
