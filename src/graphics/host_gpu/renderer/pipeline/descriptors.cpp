#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/addressBindingReport.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
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

#include <unordered_set>
#include <mutex>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <unordered_map>
#include <atomic>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

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
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

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
// whole range GPU-written, as before). A live switch.
static bool BoundedBufferWrites() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_BOUNDED_BUFFER_WRITES", 1);
	return enabled.load(std::memory_order_relaxed) != 0;
}

// Diagnostics: names the shader whose bindings are prepared, for KYTY_WATCH_GPU_WRITE.
struct DiagShaderScope {
	explicit DiagShaderScope(uint64_t hash) { BufferCache::s_diag_shader_hash = hash; }
	~DiagShaderScope() { BufferCache::s_diag_shader_hash = 0; }
	KYTY_CLASS_NO_COPY(DiagShaderScope);
};

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource&    resource,
                    const ShaderRecompiler::IR::BufferWriteExtent* extent, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	const bool metadata_read = resource.read && Config::SupportZmaskEnabled() &&
	                           (address & 0x7fffu) == 0 &&
	                           context.GetTextureCache().IsMeta(address);
	// The stores' addresses bound the bytes this draw or dispatch writes: only those become
	// GPU-written, so the guest's accesses elsewhere in the range do not drain the GPU.
	const bool bounded_write = resource.written && !metadata_read && extent != nullptr &&
	                           extent->valid && BoundedBufferWrites();
	const auto written_begin = bounded_write ? std::min(extent->begin, size) : 0u;
	const auto written_end   = bounded_write ? std::min(extent->end, size) : size;
	auto [buffer, offset] =
	    bounded_write
	        ? context.GetBufferCache().ObtainBufferWritten(address, size, address + written_begin,
	                                                       written_end - std::min(written_begin,
	                                                                              written_end),
	                                                       id)
	        : context.GetBufferCache().ObtainBuffer(
	              address, size, resource.written || metadata_read,
	              resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	if (metadata_read) buffer->Fill(offset, size, 0xf); // HTILE ZMask::kUncompressed.
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
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
			    !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
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
	// A storage read ignores the descriptor swizzle and returns the image's components with missing ones filled as
	// (0, 0, 1): on a 1- or 2-component format, (X,0,0,1) / (X,Y,0,1) reads exactly the same as identity.
	const auto identity_for_components = [&]() {
		using F = Prospero::BufferFormat;
		switch (descriptor.Format()) {
			case F::k8UNorm: case F::k8SNorm: case F::k8UInt: case F::k8SInt: case F::k16UNorm: case F::k16SNorm:
			case F::k16UInt: case F::k16SInt: case F::k16Float: case F::k32UInt: case F::k32SInt: case F::k32Float:
				return swizzle == DstSel(4, 0, 0, 1);
			case F::k8_8UNorm: case F::k8_8SNorm: case F::k8_8UInt: case F::k8_8SInt: case F::k16_16UNorm:
			case F::k16_16SNorm: case F::k16_16UInt: case F::k16_16SInt: case F::k16_16Float: case F::k32_32UInt:
			case F::k32_32SInt: case F::k32_32Float:
				return swizzle == DstSel(4, 5, 0, 1);
			default: return false;
		}
	};
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic || identity_for_components());
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
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
	       (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected;
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = uint_resource &&
	    ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	// R32 image atomics go through an R32Uint view, whatever the 32-bit format.
	const bool raw_atomic_storage = resource.atomic && !resource.atomic64 && uint_resource &&
	                                (format == Prospero::BufferFormat::k32UInt ||
	                                 format == Prospero::BufferFormat::k32SInt ||
	                                 format == Prospero::BufferFormat::k32Float);
	const bool format_ok =
	    raw_sint_storage || raw_atomic_storage ||
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

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical        = info;
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

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value) {
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
		return {.layout = vk::ImageLayout::eGeneral};
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
	const auto format         = descriptor.Format();
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
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && resource.atomic ? vk::Format::eR32Uint
	                                 : raw_sint_format != Prospero::BufferFormat::kInvalid
	                                     ? VulkanFormat(raw_sint_format)
	                                     : SrgbStorageViewFormat(pixel_format);
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
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	} else {
		desc.info.UpdateSize();
	}
	const auto alignment = desc.info.DataAlignment();
	EXIT_NOT_IMPLEMENTED(desc.info.data.size == 0 || alignment == 0 ||
	                     (address & (static_cast<uint64_t>(alignment) - 1u)) != 0);
	if (storage) {
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

	auto       id                  = texture_cache.FindImage(desc, shader_conversion);
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
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
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
static bool BindlessCompatible(const ShaderTextureResource& descriptor, uint32_t binding) {
	if (descriptor.IsNull() ||
	    Prospero::SampledTextureNumericClass(descriptor.Format()) !=
	        Prospero::TextureNumericClass::Float) {
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
	const auto address = heap.base + heap.table_offset + static_cast<uint64_t>(key) * 32u;
	heap.settled[key]  = 1;
	heap.descriptors[key] = {};
	if (!Libs::LibKernel::Memory::TryReadBacking(address, value.dwords.data(), 32u)) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	std::copy_n(value.dwords.data(), 8, heap.descriptors[key].begin());
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	if (!BindlessCompatible(descriptor, heap.binding)) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	auto binding = ResolveTexture(heap.resource, value);
	auto& texture_cache = m_context.GetTextureCache();
	auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty()) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	BindImage(binding.image_id, false);
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
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
	heap.resolved.push_back(binding.image_id);
	table.AddImageReference(binding.image_id, heap, key);
	table.SetTranslation(heap, key, slot);
	return true;
}

uint32_t RenderExecutor::RevalidateBindlessKeys(uint32_t budget) {
	// From Senaxx 0563e10e. The guest rewrites heap entries as it streams textures out and others
	// in. A key settled from the old T# kept sampling the old texture and kept it pinned in the
	// texture cache for the rest of the run (Senaxx: 14-21 keys, 44-65 MB in the jungle). Each
	// frame a window of every heap is read in one go and compared; a changed key is settled again
	// into a new slot, and an image no key refers to any more is unpinned so the cache can collect
	// it. A heap of 64 Ki keys is covered in 32 frames.
	constexpr uint32_t Window        = 2048;
	auto&              table         = m_context.GetBindlessTable();
	auto&              texture_cache = m_context.GetTextureCache();
	uint32_t           resolved      = 0;
	for (auto& heap: table.Heaps()) {
		const auto keys = static_cast<uint32_t>(heap.settled.size());
		if (keys == 0) {
			continue;
		}
		const uint32_t begin = heap.revalidate_cursor < keys ? heap.revalidate_cursor : 0u;
		const uint32_t count = std::min(Window, keys - begin);
		m_bindless_window.resize(count);
		const auto address  = heap.base + heap.table_offset + static_cast<uint64_t>(begin) * 32u;
		const bool readable = Libs::LibKernel::Memory::TryReadBacking(
		    address, m_bindless_window.data(), static_cast<uint64_t>(count) * 32u);
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
			const auto old_image = heap.images[key];
			if (table.ReleaseKey(heap, key)) {
				if (auto* image = texture_cache.m_slot_images.try_get(old_image);
				    image != nullptr && image->bindless_pinned) {
					NoteBindlessStateChange(*image);
					image->bindless_pinned = false;
				}
			}
			resolved += ResolveBindlessKey(heap, key) ? 1u : 0u;
		}
		heap.revalidate_cursor = next >= keys ? 0u : next;
	}
	return resolved;
}

void RenderExecutor::ResolveBindlessRequests() {
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
	m_context.GetBindlessTable().DrainWatchdogReports(frame);
	if (!table.SnapshotRecorded()) {
		table.RecordFeedbackSnapshot(scheduler);
		return;
	}
	table.ConsumeSnapshot();
	// Each texture may upload and detile; spread first sight of a scene over a few frames.
	constexpr uint32_t Budget   = 128;
	// KYTY_BINDLESS_REVALIDATE=0 (live, default 1) leaves changed heap entries settled.
	static auto& revalidate = Common::LiveSwitches::Get("KYTY_BINDLESS_REVALIDATE", 1);
	uint32_t resolved =
	    revalidate.load(std::memory_order_relaxed) != 0 ? RevalidateBindlessKeys(Budget) : 0u;
	m_bindless_used.clear();
	for (auto& heap: m_context.GetBindlessTable().Heaps()) {
		m_bindless_requests.clear();
		m_context.GetBindlessTable().TakeRequests(heap, m_bindless_requests, m_bindless_used);
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
	// Shaders sampled these since the last snapshot; draws never bind them, so this is the only
	// place their age in the texture cache is refreshed.
	m_context.GetTextureCache().TouchImages(m_bindless_used);
	table.RecordFeedbackSnapshot(scheduler);
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
		std::ranges::copy(ShaderRecompiler::IR::FallbackSampler(), default_sampler.fields);
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
		// KYTY_BINDLESS_SAMPLER_SLOTS_OFF=1 (live, research): patch an empty range, so every key
		// selects slot 0 (the default sampler) while the rest of the bindless sampler path stays.
		static auto& slots_off = Common::LiveSwitches::Get("KYTY_BINDLESS_SAMPLER_SLOTS_OFF", 0);
		if (slots_off.load(std::memory_order_relaxed) != 0) {
			region  = 0;
			entries = 0;
		}
		// KYTY_BINDLESS_SAMPLER_KEY_SHIFT=N (live, research): key k selects record k + N. Run 72:
		// terrain gets a wrong record (its records alternate clamp/wrap; a clamped tiling texture
		// is one flat colour); this tests an off-by-N table offset.
		static auto& key_shift = Common::LiveSwitches::Get("KYTY_BINDLESS_SAMPLER_KEY_SHIFT", 0);
		// KYTY_KEY_SHIFT_SKIP_BITS=k / KYTY_KEY_SHIFT_SKIP_VALUE=v (live, research): the shift
		// skips shaders whose hash has v in its low k bits, to find by bisection which shaders
		// need the unshifted records (Wolverine's moon).
		static auto& skip_bits  = Common::LiveSwitches::Get("KYTY_KEY_SHIFT_SKIP_BITS", 0);
		static auto& skip_value = Common::LiveSwitches::Get("KYTY_KEY_SHIFT_SKIP_VALUE", 0);
		bool         skip_shift = false;
		if (const auto bits = skip_bits.load(std::memory_order_relaxed); bits > 0 && bits < 64) {
			const uint64_t mask = (uint64_t {1} << static_cast<uint32_t>(bits)) - 1u;
			skip_shift          = (program.shader_hash & mask) ==
			             (static_cast<uint64_t>(skip_value.load(std::memory_order_relaxed)) & mask);
		}
		{
			static auto&                        dump_hashes = Common::LiveSwitches::Get(
                "KYTY_BINDLESS_SAMPLER_SHADERS", 0);
			static std::unordered_set<uint64_t> printed;
			if (dump_hashes.load(std::memory_order_relaxed) != 0 &&
			    printed.insert(program.shader_hash).second) {
				::printf("Bindless sampler shader 0x%016llx\n",
				         static_cast<unsigned long long>(program.shader_hash));
				std::fflush(stdout);
			}
		}
		if (const auto shift = key_shift.load(std::memory_order_relaxed);
		    shift != 0 && !skip_shift && region != 0 && entries != 0) {
			const auto magnitude = static_cast<uint32_t>(shift > 0 ? shift : -shift);
			if (shift > 0 && magnitude < entries) {
				region += magnitude;
				entries -= magnitude;
			} else if (shift < 0 && magnitude < region) {
				region -= magnitude;
			}
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
	if ((snapshot.bindless_heaps.empty() && snapshot.bindless_sampler_heaps.empty()) ||
	    !table.Enabled()) {
		return;
	}
	PrepareBindlessSamplers(runtime, prepared);
	if (snapshot.bindless_heaps.empty()) {
		return;
	}
	ResolveBindlessRequests();
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.image >= program.info.images.size()) {
			continue;
		}
		const auto& resource = program.info.images[use.image];
		const uint64_t span  = use.size > use.table_offset ? use.size - use.table_offset : 0;
		const auto entries   = static_cast<uint32_t>(std::min<uint64_t>(span / 32u, 1u << 18u));
		auto* heap = table.FindOrCreateHeap(use.base, use.table_offset, BindlessBindingFor(resource),
		                                    entries, resource);
		if (heap == nullptr) {
			continue; // region 0, count 0: every key samples the placeholder
		}
		prepared.bindless_patches.push_back(
		    {use.mapping_offset, heap->region, std::min(entries, heap->entries)});
		prepared.bindless_heaps.push_back(heap);
	}
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	PrepareBindlessHeaps(runtime, prepared);
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
		EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
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
			// Descriptors in guest page 0 bind as null.
			if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
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
				} else {
					const auto& graphics = m_context.GetGraphics();
					const auto limit = uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
					                   (graphics.StorageMinAlignment() - 1);
					size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
				}
			}
			// Research: a record count only the shader knows binds a capped range; the shader
			// applies the real bound itself.
			constexpr uint64_t GpuRecordsBinding = 64ull << 20u;
			const auto requested_size = metadata.gpu_records ? std::min(size, GpuRecordsBinding) : size;
			size = Libs::LibKernel::Memory::TryClampRangeSize(address, requested_size);
			if (size == 0) {
				// Nothing is mapped at the base yet. Streaming titles bind regions they back later (a
				// dispatch can precede the mapping by thousands of calls), and an unbacked region reads
				// zero and drops writes on the hardware; binding null gives the same.
				static std::atomic<uint32_t> reported = 0;
				if (reported.fetch_add(1) < 8) {
					LOGF("RenderExecutor: buffer descriptor %u outside mapped memory: stage=%u "
					     "base=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
					     resource, static_cast<uint32_t>(program.stage), address, requested_size);
				}
				source = {};
				continue;
			}
			source = {address, size, cache.FindBuffer(address, size)};
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.MemoryOffsetCount());
	// Diagnostics name this shader as the writer of the buffers bound below (KYTY_WATCH_GPU_WRITE,
	// KYTY_READBACK_STATS); the shader looked up last can be another stage's or dispatch's.
	DiagShaderScope diag(program.shader_hash);
	BufferCache::SetReadbackStatsShader(program.shader_hash);

	prepared.buffers.resize(layout.MemoryOffsetCount());
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (uint32_t resource_index = 0; resource_index < program.info.buffers.size();
	     ++resource_index) {
		const auto& resource = program.info.buffers[resource_index];
		const auto  index    = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		prepared.buffers[index] = NativeStorageBuffer(
		    m_context, prepared.buffer_sources[index], resource,
		    resource_index < snapshot.buffer_write_extents.size()
		        ? &snapshot.buffer_write_extents[resource_index]
		        : nullptr,
		    buffer_offset);
		pack_memory_offset(index, buffer_offset);
		// The fallback allocation makes the Vulkan descriptor valid, but an empty guest
		// buffer still has no accessible elements. In particular, never let a null store
		// modify the shared fallback and affect subsequent null reads.
		prepared.shader_data[layout.BufferLengthDword() + index] =
		    prepared.buffer_sources[index].size == 0
		        ? 0u
		        : static_cast<uint32_t>(prepared.buffers[index].range / sizeof(uint32_t));
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		if (prepared.bindless_patches.empty()) {
			prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
		} else {
			m_bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
			for (const auto& [offset, region, entries]: prepared.bindless_patches) {
				if (static_cast<size_t>(offset) + 1u < m_bindless_srt.size()) {
					m_bindless_srt[offset]      = region;
					m_bindless_srt[offset + 1u] = entries;
				}
			}
			prepared.flattened_srt = NativeUpload(m_context, m_bindless_srt);
		}
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || !old_image->registered ||
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
		// A deleted image (unregistered, its slot erased once the GPU passes the deletion tick) must
		// not be bound: a later command buffer would read it after its memory is gone.
		if (!image.registered && !image.info.data.Empty()) {
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 32) {
				std::printf("Binding a deleted image: shader 0x%016llx img%u guest 0x%" PRIx64
				            " %ux%u layers %u image %p\n",
				            static_cast<unsigned long long>(program.shader_hash), i,
				            image.info.data.address, image.info.extent.width,
				            image.info.extent.height, image.info.resources.layers,
				            static_cast<void*>(static_cast<VkImage>(image.backing.image)));
				std::fflush(stdout);
			}
		}
		// KYTY_TRACE_PS=<hash> (environment, research): prints each change of the images a
		// program binds (guest range, size, Vulkan image, registered), to follow a read of a
		// freed image back to the binding that handed it out (Wolverine device losts, run 92).
		static const uint64_t trace_hash = [] {
			const char* value = std::getenv("KYTY_TRACE_PS");
			return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
		}();
		if (trace_hash != 0 && program.shader_hash == trace_hash) {
			static std::unordered_map<uint64_t, VkImage> last;
			const auto vk_image = static_cast<VkImage>(image.backing.image);
			auto&      previous = last[(static_cast<uint64_t>(i) << 56u) ^ program.shader_hash];
			if (previous != vk_image) {
				previous = vk_image;
				std::printf("Trace PS 0x%016llx img%u: guest 0x%" PRIx64 " size 0x%" PRIx64
				            " %ux%ux%u layers %u fmt %u image %p registered %d tick %" PRIu64 "\n",
				            static_cast<unsigned long long>(program.shader_hash), i,
				            image.info.data.address, image.info.data.size, image.info.extent.width,
				            image.info.extent.height, image.info.extent.depth, image.info.resources.layers,
				            static_cast<uint32_t>(image.info.pixel_format),
				            static_cast<void*>(vk_image), image.registered ? 1 : 0,
				            m_context.GetCommandScheduler().CurrentTick());
				std::fflush(stdout);
			}
		}
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma = false;
	FindBuffers(stages);
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->program->info.uses_dma;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || !old_image->registered ||
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
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	const auto vk_buffer              = buffer.Recorder();
	size_t     image_descriptor_count = 0;
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
	// The host evaluated these scalar reads (branch inputs, descriptor selections) from memory as
	// it was before the draw or dispatch. Upstream (6409be28) exits when a bound shader may write
	// them; Wolverine does this (CS 0x85a58319a7a75e48 reads a branch input from a buffer the same
	// dispatch updates) and renders correctly with the pre-dispatch values, as the walk did before
	// this check (KYTY_IGNORE_WRITTEN_OVERLAP=1). Each reader/writer pair is reported once.
	const auto report_overlap = [](const char* what, uint64_t reader_hash, uint64_t writer_hash,
	                               uint64_t address, uint64_t size, uint64_t written_address,
	                               uint64_t written_size) {
		static std::mutex                   reported_mutex;
		static std::unordered_set<uint64_t> reported;
		std::scoped_lock                    lock(reported_mutex);
		if (reported.insert(reader_hash ^ (writer_hash << 1u) ^ reinterpret_cast<uintptr_t>(what))
		        .second) {
			LOGF("Warning: scalar resource reads overlap %s: reader 0x%016" PRIx64 " reads 0x%" PRIx64
			     "+0x%" PRIx64 ", writer 0x%016" PRIx64 " writes 0x%" PRIx64 "+0x%" PRIx64 "\n",
			     what, reader_hash, address, size, writer_hash, written_address, written_size);
		}
	};
	for (const auto* reader: prepared_bindings) {
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		const auto reader_hash = reader->runtime->program->shader_hash;
		for (const auto* writer: prepared_bindings) {
			if (writer->runtime->program->has_address_writes) {
				report_overlap("shader address writes (unprovable)", reader_hash,
				               writer->runtime->program->shader_hash, 0, 0, 0, 0);
			}
		}
		for (const auto [address, size]: reads) {
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0 && ImageRangeOverlaps(address, size,
					                                          written.address, written.size)) {
						report_overlap("an image or attachment write", reader_hash, 0, address, size,
						               written.address, written.size);
					}
				}
			}
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				const auto& extents = writer->runtime->resources->buffer_write_extents;
				for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
					const auto& metadata = program.info.buffers[resource];
					if (!metadata.written || metadata.descriptor_index == UINT32_MAX) continue;
					const auto& written = writer->buffer_sources[metadata.descriptor_index];
					// The bytes the dispatch can store to, when its stores are bounded: a heap-wide
					// V# otherwise covers every descriptor read in the heap.
					auto written_address = written.address;
					auto written_size    = written.size;
					if (resource < extents.size() && extents[resource].valid) {
						written_address += extents[resource].begin;
						written_size = extents[resource].end > extents[resource].begin
						                   ? extents[resource].end - extents[resource].begin
						                   : 0;
					}
					if (written_size != 0 &&
					    ImageRangeOverlaps(address, size, written_address, written_size)) {
						report_overlap("a shader buffer write", reader_hash, program.shader_hash,
						               address, size, written_address, written_size);
					}
				}
			}
		}
	}
	m_descriptor_images.resize(image_descriptor_count);
	// Device-loss triage: the images each draw binds (KYTY address binding report).
	const bool note_bindings = m_context.GetGraphics().address_binding_report_enabled;
	m_descriptor_writes.clear();
	m_descriptor_writes.reserve(write_count);
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;

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
		for (auto* heap: descriptors.bindless_heaps) {
			const auto& resolved = heap->resolved;
			size_t      begin =
			    heap->checked_generation ==
			            g_bindless_state_generation.load(std::memory_order_relaxed) &&
			            heap->checked_count <= resolved.size()
			         ? heap->checked_count
			         : 0;
			// KYTY_BINDLESS_SKIP=0 (live) checks every image again, for same-process A/B.
			static auto& skip = Common::LiveSwitches::Get("KYTY_BINDLESS_SKIP", 1);
			if (verify || skip.load(std::memory_order_relaxed) == 0) {
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
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			}
			// After the transitions above, which bump the generation themselves.
			heap->checked_generation = g_bindless_state_generation.load(std::memory_order_relaxed);
			heap->checked_count      = resolved.size();
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& binding = descriptors.images[i];
			if (!binding.image_id) continue;
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
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
						EXIT_IF(descriptors.buffers.size() != write.descriptorCount);
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
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							EXIT_IF(sampler == nullptr);
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
			EXIT_IF(texture.image_id &&
			        resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()));
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
			if (note_bindings && texture.image_id) {
				if (const auto* image =
				        m_context.GetTextureCache().m_slot_images.try_get(texture.image_id);
				    image != nullptr) {
					AddressBindingNoteDescriptor(
					    reinterpret_cast<uint64_t>(static_cast<VkImage>(image->backing.image)),
					    program.shader_hash);
				}
			}
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
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
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
