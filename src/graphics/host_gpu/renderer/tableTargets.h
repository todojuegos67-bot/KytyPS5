#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLETARGETS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLETARGETS_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

// KYTY_TABLE (tableMode.h): the render targets' table. Per color slot and for depth, the last
// few register states resolved with an image found with the same backing, and what they resolved
// to; an entry holds while the texture cache's image set generation is the one after its lookup.
// GPU thread only (RenderExecutor).
namespace Libs::Graphics::TableTargets {

constexpr uint32_t Ways = 4;

struct ColorEntry {
	bool             valid      = false;
	uint64_t         generation = 0;
	HW::RenderTarget rt {};
	uint32_t         mask         = 0;
	uint32_t         slice_offset = 0;
	bool             exact_format = false;
	RenderColorInfo  result;
};

struct DepthEntry {
	bool                    valid      = false;
	uint64_t                generation = 0;
	HW::DepthRenderTarget   z {};
	HW::RenderControl       rc {};
	HW::DepthControl        dc {};
	HW::StencilControl      sc {};
	HW::StencilMask         sm {};
	float                   depth_clear = 0.0f;
	float                   bounds_min  = 0.0f;
	float                   bounds_max  = 0.0f;
	uint8_t                 stencil_clear = 0;
	RenderDepthInfo         result;
};

template <typename T>
[[nodiscard]] inline bool SameBytes(const T& a, const T& b) {
	static_assert(std::is_trivially_copyable_v<T>);
	return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
inline void CopyBytes(T& to, const T& from) {
	static_assert(std::is_trivially_copyable_v<T>);
	std::memcpy(&to, &from, sizeof(T));
}

struct Table {
	std::array<std::array<ColorEntry, Ways>, 8> color {};
	std::array<uint32_t, 8>                     color_next {};
	std::array<DepthEntry, Ways>                depth {};
	uint32_t                                    depth_next = 0;
};

// KYTY_TABLE=2: what the verifier compares of a target's description.
[[nodiscard]] inline bool SameDesc(const TextureCache::ImageDesc& a,
                                   const TextureCache::ImageDesc& b) {
	const auto& x = a.info;
	const auto& y = b.info;
	return a.type == b.type && a.view_info == b.view_info &&
	       x.data.address == y.data.address && x.data.size == y.data.size &&
	       x.stencil.address == y.stencil.address && x.stencil.size == y.stencil.size &&
	       x.metadata.kind == y.metadata.kind &&
	       x.metadata.range.address == y.metadata.range.address &&
	       x.metadata.range.size == y.metadata.range.size &&
	       x.metadata.clear_word == y.metadata.clear_word &&
	       x.metadata.clear_word1 == y.metadata.clear_word1 &&
	       x.pixel_format == y.pixel_format && x.guest_format == y.guest_format &&
	       x.type == y.type && x.extent == y.extent && x.resources == y.resources &&
	       x.pitch == y.pitch && x.bytes_per_block == y.bytes_per_block &&
	       x.samples == y.samples && x.tile_mode == y.tile_mode && x.mip_layout == y.mip_layout;
}

} // namespace Libs::Graphics::TableTargets

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TABLETARGETS_H_
