#version 450

// KYTY_FRAMEGEN_BACKEND=fsr (frame-gen.cpp, RecordHudless): FSR's HUD-less image, written over a copy of the
// backbuffer: where the game's interface canvas covers the game image (its alpha, a few pixels around), the HUD is
// replaced by the scene around it (the nearest ring of samples outside the HUD, ever wider). FSR interpolates that
// image (no HUD left to drag along with the scene) and takes the pixels that differ from the backbuffer - the HUD's -
// from the game's frame.

layout(location = 0) in vec2 uv_unused;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D canvas; // the interface canvas (its alpha: the HUD)
layout(set = 0, binding = 1) uniform sampler2D source; // the backbuffer, copied

layout(push_constant) uniform Parameters {
	vec4  region;    // the game image in the backbuffer: offset x, y, width, height (pixels)
	vec2  extent;    // the backbuffer's size (pixels)
	float threshold; // canvas alpha above which a pixel is HUD
	float invert;    // 1: the canvas's alpha is the scene's share (1 - coverage)
} p;

float Coverage(vec2 uv) {
	float a = texture(canvas, uv).a;
	return p.invert > 0.5 ? 1.0 - a : a;
}

// The pixel at `pixel` (backbuffer pixels) is HUD: the canvas covers it or a pixel near it.
bool Hud(vec2 pixel) {
	vec2 uv = (pixel - p.region.xy) / p.region.zw;
	if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return false;
	vec2  d = 2.5 / p.region.zw;
	float a = max(max(Coverage(uv), max(Coverage(uv + vec2(d.x, 0.0)), Coverage(uv - vec2(d.x, 0.0)))),
	              max(max(Coverage(uv + vec2(0.0, d.y)), Coverage(uv - vec2(0.0, d.y))),
	                  max(Coverage(uv + d), Coverage(uv - d))));
	return a > p.threshold;
}

const vec2 Directions[8] = vec2[8](vec2(1.0, 0.0), vec2(0.7071, 0.7071), vec2(0.0, 1.0), vec2(-0.7071, 0.7071),
                                   vec2(-1.0, 0.0), vec2(-0.7071, -0.7071), vec2(0.0, -1.0), vec2(0.7071, -0.7071));

void main() {
	vec2 pixel = gl_FragCoord.xy;
	// (The copy as it is, written rather than discarded: at Vulkan 1.3 a discard is a demote, a device feature.)
	if (!Hud(pixel)) {
		color = vec4(texture(source, pixel / p.extent).rgb, 1.0);
		return;
	}
	vec3  sum    = vec3(0.0);
	float weight = 0.0;
	for (float radius = 4.0; radius <= 512.0 && weight < 3.0; radius *= 2.0) {
		for (int i = 0; i < 8; ++i) {
			vec2 at = pixel + Directions[i] * radius;
			if (any(lessThan(at, vec2(0.5))) || any(greaterThan(at, p.extent - 0.5)) || Hud(at)) continue;
			sum += texture(source, at / p.extent).rgb;
			weight += 1.0;
		}
	}
	color = vec4(weight > 0.0 ? sum / weight : texture(source, pixel / p.extent).rgb * 0.5, 1.0);
}
