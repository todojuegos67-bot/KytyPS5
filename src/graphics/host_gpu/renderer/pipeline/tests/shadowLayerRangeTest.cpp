// Standalone test of shadowLayerRange.h (not in the CMake build): clang++ -std=c++20 -Isrc <this file> && ./a.out
#include "graphics/host_gpu/renderer/pipeline/shadowLayerRange.h"
#include <cassert>
#include <cstdio>
using namespace Libs::Graphics::ShadowLayerRange;
int main() {
	// The crash case: BC7 2353 layers of 1 MiB, base 754, shader bound 256, GPU-written overlap.
	Input in; in.image_layers = 2353; in.base_layer = 754; in.bound_count = 256; in.layer_bytes = 1u << 20; in.gpu_written = true;
	auto r = ViewLayers(in); assert(r.layer_count == 256 && r.capped);
	in.fix = false; r = ViewLayers(in); assert(r.layer_count == 1599 && !r.capped); // old behaviour
	// Spot-shadow pool: 80 R32 layers of 4 MiB, GPU-written, bound smaller -> whole (as before).
	Input s; s.image_layers = 80; s.base_layer = 0; s.bound_count = 4; s.layer_bytes = 4u << 20; s.gpu_written = true;
	r = ViewLayers(s); assert(r.layer_count == 80 && !r.capped);
	s.fix = false; assert(ViewLayers(s).layer_count == 80);
	// Depth-compare of a huge array: unchanged (whole).
	Input d = in; d.fix = true; d.gpu_written = false; d.shadow_sample = true; assert(ViewLayers(d).layer_count == 1599);
	// Not shadow, not GPU-written: the bound.
	Input n = in; n.fix = true; n.gpu_written = false; assert(ViewLayers(n).layer_count == 256 && !ViewLayers(n).capped);
	// Bytes cap: 200 layers of 4 MiB = 800 MiB > 512 -> bound.
	Input b = s; b.fix = true; b.image_layers = 200; assert(ViewLayers(b).layer_count == 4);
	// Huge T#, base near its end: few layers left -> whole range is fine.
	Input e = in; e.fix = true; e.base_layer = 2300; e.bound_count = 53; assert(ViewLayers(e).layer_count == 53);
	// base out of range / single layer: the bound.
	Input o = in; o.fix = true; o.base_layer = 2353; assert(ViewLayers(o).layer_count == 256);
	std::puts("ok");
}
