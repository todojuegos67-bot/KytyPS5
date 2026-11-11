#version 450

// The presenter's copy of a 10-bit (or half-float) guest frame into an 8-bit swapchain image, with an ordered dither of
// up to half an 8-bit step: the blit it replaces rounded every pixel, and smooth dark gradients that move (the loading
// screens' fog) showed their 8-bit bands crawling (issue #234). The same scaling as that blit (a linear filter).

layout(set = 0, binding = 0) uniform sampler2D source_image;

layout(push_constant) uniform Constants {
	vec2 offset; // the destination rectangle's top left, in pixels
	vec2 scale;  // source texture coordinates per destination pixel
} constants;

layout(location = 0) out vec4 color;

void main() {
	vec2 uv    = (gl_FragCoord.xy - constants.offset) * constants.scale;
	vec4 pixel = textureLod(source_image, uv, 0.0);
	// A 4x4 Bayer matrix, the same pattern every frame (a still picture stays still): (rank + 0.5) / 16 - 0.5 of a step.
	const float bayer[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
	uvec2 cell   = uvec2(gl_FragCoord.xy) & 3u;
	float dither = (bayer[cell.y * 4u + cell.x] + 0.5) / 16.0 - 0.5;
	color        = vec4(pixel.rgb + dither / 255.0, pixel.a);
}
