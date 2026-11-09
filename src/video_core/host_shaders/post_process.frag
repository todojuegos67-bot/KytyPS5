// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;

layout (push_constant) uniform settings {
    float gamma;
    bool hdr;
    bool srgb_input;
    float downscale;
} pp;

float catrom(float x) {
    x = abs(x);
    if (x < 1.0) return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    if (x < 2.0) return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    return 0.0;
}

// Larger input than the output (an upscaler output above the window): Catmull-Rom widened by
// the ratio, so the downscale neither aliases nor blurs like bilinear.
vec4 downsample(vec2 st) {
    const ivec2 size = textureSize(texSampler, 0);
    const vec2 p = st * vec2(size) - 0.5;
    const float s = pp.downscale;
    const int r = int(ceil(2.0 * s));
    const ivec2 c = ivec2(floor(p));
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int j = -r + 1; j <= r; ++j) {
        const float wy = catrom((float(c.y + j) - p.y) / s);
        if (wy == 0.0) continue;
        for (int i = -r + 1; i <= r; ++i) {
            const float w = wy * catrom((float(c.x + i) - p.x) / s);
            if (w == 0.0) continue;
            sum += w * texelFetch(texSampler, clamp(c + ivec2(i, j), ivec2(0), size - 1), 0);
            wsum += w;
        }
    }
    return max(sum / wsum, vec4(0.0));
}

const float cutoff = 0.0031308, a = 1.055, b = 0.055, d = 12.92;
vec3 gamma(vec3 rgb) {
    return mix(
        a * pow(rgb, vec3(1.0 / (2.4 + 1.0 - pp.gamma))) - b,
        d * rgb / pp.gamma,
        lessThan(rgb, vec3(cutoff))
    );
}

// Exact inverse of gamma() at unit gamma, for buffers that are sRGB encoded but must be sampled
// through a UNORM view because Vulkan has no sRGB variant of the 10-bit format.
vec3 degamma(vec3 rgb) {
    return mix(
        pow(max(rgb + b, 0.0) / a, vec3(2.4)),
        rgb / d,
        lessThan(rgb, vec3(d * cutoff))
    );
}

void main() {
    vec4 color_linear = pp.downscale > 1.01 ? downsample(uv) : texture(texSampler, uv);
    if (pp.hdr) {
        color = color_linear;
    } else {
        if (pp.srgb_input) color_linear.rgb = degamma(color_linear.rgb);
        color = vec4(gamma(color_linear.rgb), color_linear.a);
    }
}
