#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADOWLAYERRANGE_H_
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADOWLAYERRANGE_H_

#include <cstdint>

namespace Libs::Graphics::ShadowLayerRange {

// How many layers a sampled array view of a "shadow-map array" holds (descriptors.cpp, 1e1804141): a depth-compare
// sample or a depth image keeps every layer from the T#'s base to its last; so did any array whose memory the GPU
// wrote, as the 80-layer spot-shadow pool (R32, 4 MiB a layer: 320 MiB). But the game's streamed 1024x1024 texture
// pool views are T#s of thousands of layers (0x1021400000: BC7, 2353 layers of 1 MiB, base 754, a shader bound of 256)
// over memory the GPU also renders into: kept whole they were a 1599-layer (1.6 GiB) image, one allocation and a 262 ms
// upload, before a crash on a 16 GB PC. With `fix` (KYTY_LAYER_RANGE_FIX, default on) an array kept whole only for the
// GPU-written memory keeps the shader's bound when its layers from the base pass `max_layers` or `max_bytes`.
struct Input {
	uint32_t image_layers   = 0;     // the T#'s layers (DEPTH + 1)
	uint32_t base_layer     = 0;     // the T#'s BASE_ARRAY
	uint32_t bound_count    = 0;     // the view's layers solved from the shader (ArrayViewLayers)
	uint64_t layer_bytes    = 0;     // the guest bytes of one layer (all its levels)
	bool     shadow_sample  = false; // a depth-compare sample or a depth image
	bool     gpu_written    = false; // memory the GPU wrote, or under a GPU-written image
	bool     fix            = true;
	uint32_t max_layers     = 256;
	uint64_t max_bytes      = uint64_t {512} << 20u;
};

struct Result {
	uint32_t layer_count = 0;
	bool     capped      = false; // kept to the shader's bound where it would have been kept whole
};

inline Result ViewLayers(const Input& in) {
	Result r {in.bound_count, false};
	if (in.image_layers <= 1 || in.base_layer >= in.image_layers || !(in.shadow_sample || in.gpu_written)) {
		return r;
	}
	const uint32_t full = in.image_layers - in.base_layer;
	if (in.fix && !in.shadow_sample &&
	    (full > in.max_layers || static_cast<uint64_t>(full) * in.layer_bytes > in.max_bytes)) {
		r.capped = full > in.bound_count;
		return r;
	}
	r.layer_count = full;
	return r;
}

} // namespace Libs::Graphics::ShadowLayerRange

#endif /* GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADOWLAYERRANGE_H_ */
