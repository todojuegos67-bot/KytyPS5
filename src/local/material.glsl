// The material decode shared by the passes that read Demon's Souls' packed G-buffer copy (4 x R32_UINT layers taken
// before the deferred lighting, frame-gen.cpp SsrCaptureGbuffer). The codes are rt-reflections' (rt-reflections.comp
// keeps its own copy of the normal layouts; the numbering is the same) and material.cpp's:
//   normal: layer | layout << 2 | 64 (world space) | 128 (negated); layouts 0..12 as DecodeNormal below
//   field:  layer | shift << 2 | bits << 7 | 4096 (inverted: 1 - v), a value in 0..1
//   albedo: layer | shift << 2 | 64 (reversed byte order: b, g, r) | 128 (linear, no sRGB decode): three 8-bit bytes
// The includer defines `uint Packed(int layer, ivec2 p)` first (the layer's word at a render pixel).

float MatBits(uint w, int shift, int bits) {
	return float((w >> uint(shift)) & ((1u << uint(bits)) - 1u)) / float((1u << uint(bits)) - 1u);
}
vec3 MatUnit(vec3 n) {
	float l = length(n);
	return l > 1e-4 ? n / l : vec3(0, 0, 1);
}
vec3 MatOct(vec2 e) {
	e       = e * 2.0 - 1.0;
	vec3  n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
	float t = max(-n.z, 0.0);
	n.x += n.x >= 0.0 ? -t : t;
	n.y += n.y >= 0.0 ? -t : t;
	return normalize(n);
}
float MatSnorm16(uint w, int shift) {
	int v = int(w << uint(16 - shift)) >> 16;
	return clamp(float(v) / 32767.0, -1.0, 1.0);
}
// 0 oct 16:16, 1 oct 12:12 bits 0..23, 2 oct 12:12 bits 8..31, 3 oct 11:11, 4 xyz 10:10:10, 5 oct 16:16 swapped,
// 6 oct 16:16 snorm, 7 spheremap 16:16, 8 xyz 11:11:10, 9 oct 8:8 bits 0..15, 10 oct 8:8 bits 16..31, 11 hemi-oct 16:16,
// 12 xyz 10:10:10 bits 2..31 (Demon's Souls: layer 3, layout 9, world space; mean cos 0.983 against depth normals).
vec3 MatDecodeNormal(uint w, int layout_) {
	if (layout_ == 9) return MatOct(vec2(MatBits(w, 0, 8), MatBits(w, 8, 8)));
	if (layout_ == 0) return MatOct(vec2(MatBits(w, 0, 16), MatBits(w, 16, 16)));
	if (layout_ == 1) return MatOct(vec2(MatBits(w, 0, 12), MatBits(w, 12, 12)));
	if (layout_ == 2) return MatOct(vec2(MatBits(w, 8, 12), MatBits(w, 20, 12)));
	if (layout_ == 3) return MatOct(vec2(MatBits(w, 0, 11), MatBits(w, 11, 11)));
	if (layout_ == 4) return MatUnit(vec3(MatBits(w, 0, 10), MatBits(w, 10, 10), MatBits(w, 20, 10)) * 2.0 - 1.0);
	if (layout_ == 5) return MatOct(vec2(MatBits(w, 16, 16), MatBits(w, 0, 16)));
	if (layout_ == 6) return MatOct(vec2(MatSnorm16(w, 0), MatSnorm16(w, 16)) * 0.5 + 0.5);
	if (layout_ == 7) {
		vec2  f  = vec2(MatBits(w, 0, 16), MatBits(w, 16, 16)) * 4.0 - 2.0;
		float ff = dot(f, f);
		float g  = sqrt(max(1.0 - ff * 0.25, 0.0));
		return MatUnit(vec3(f * g, 1.0 - ff * 0.5));
	}
	if (layout_ == 8) return MatUnit(vec3(MatBits(w, 0, 11), MatBits(w, 11, 11), MatBits(w, 22, 10)) * 2.0 - 1.0);
	if (layout_ == 10) return MatOct(vec2(MatBits(w, 16, 8), MatBits(w, 24, 8)));
	if (layout_ == 11) {
		vec2 e = vec2(MatBits(w, 0, 16), MatBits(w, 16, 16)) * 2.0 - 1.0;
		vec2 t = vec2(e.x + e.y, e.x - e.y) * 0.5;
		return MatUnit(vec3(t, 1.0 - abs(t.x) - abs(t.y)));
	}
	return MatUnit(vec3(MatBits(w, 2, 10), MatBits(w, 12, 10), MatBits(w, 22, 10)) * 2.0 - 1.0);
}
// The normal in the space the code says (world: the includer rotates it); false when absent or degenerate.
bool MatNormalRaw(int code, ivec2 p, out vec3 n, out bool world) {
	n     = vec3(0, 0, 1);
	world = false;
	if (code < 0) return false;
	vec3 d = MatDecodeNormal(Packed(code & 3, p), (code >> 2) & 15);
	if ((code & 128) != 0) d = -d;
	float l = length(d);
	if (!(l > 1e-4)) return false;
	n     = d / l;
	world = (code & 64) != 0;
	return true;
}
float MatField(int code, ivec2 p) {
	if (code < 0) return -1.0;
	float v = MatBits(Packed(code & 3, p), (code >> 2) & 31, max((code >> 7) & 31, 1));
	return (code & 4096) != 0 ? 1.0 - v : v;
}
float MatFieldOf(int code, uint w) {
	float v = MatBits(w, (code >> 2) & 31, max((code >> 7) & 31, 1));
	return (code & 4096) != 0 ? 1.0 - v : v;
}
// Linear albedo (sRGB-encoded bytes unless flag 128); -1 when no albedo field is known.
vec3 MatAlbedo(int code, ivec2 p) {
	if (code < 0) return vec3(-1.0);
	uint w  = Packed(code & 3, p);
	int  s  = (code >> 2) & 31;
	vec3 c  = vec3(MatBits(w, s, 8), MatBits(w, s + 8, 8), MatBits(w, s + 16, 8));
	if ((code & 64) != 0) c = c.bgr;
	if ((code & 128) == 0) c = pow(c, vec3(2.2));
	return c;
}
