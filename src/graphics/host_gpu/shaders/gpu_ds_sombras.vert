#version 450
// (F5 pase 2, KYTY_NATIVO_PASE_SOMBRAS=3/=1; renderer/dsSombrasNativas.inc) El VS de profundidad de las sombras en
// cascada de Demon's Souls (ES @26c92d600, 121 dwords; notas-sombras-nativas.md §3) operacion por operacion. Cada
// v_mad/v_mac de GFX10 es una multiplicacion redondeada y luego una suma (no fusionadas): `precise` lo impone aqui.
// Camara (tabla[1]): la vista de la cascada (*(ctx+0x100), 0x720 bytes) como UBO; objeto (tabla[2], 12 float) y
// tabla[3]+0x14 (4 float) por draw en push constants.
layout(location = 0) in vec3 position; // (buffer_load_format_xyz del V# de tabla[0]+0x10: 32_32_32_FLOAT, stride 12)

layout(std140, set = 0, binding = 0) uniform Camera {
	vec4 c[114]; // (0x720 bytes)
} camera;

layout(push_constant) uniform Draw {
	vec4 o0; // s24..s27
	vec4 o1; // s28..s31
	vec4 o2; // s48..s51
	vec4 t3; // s36..s39
} draw;

out gl_PerVertex {
	vec4 gl_Position;
};

float C(int byte_offset) {
	int i = byte_offset >> 2;
	return camera.c[i >> 2][i & 3];
}

void main() {
	// s_load: +0x6e0 -> s0..s7, +0x700 -> s32..s35, +0x2d8 -> vcc_lo, +0x2d0 -> s56,s57, +0x200 -> s40..s43,
	// +0x1e0 -> s8..s15, +0x30 -> s44..s47, +0x0 -> s16..s23, +0x1c0 -> s58,s59.
	float s0 = C(0x6e0), s1 = C(0x6e4), s2 = C(0x6e8), s4 = C(0x6f0), s5 = C(0x6f4), s6 = C(0x6f8);
	float s32 = C(0x700), s33 = C(0x704), s34 = C(0x708);
	float vcc = C(0x2d8), s56 = C(0x2d0), s57 = C(0x2d4);
	float s40 = C(0x200), s41 = C(0x204), s42 = C(0x208);
	float s8 = C(0x1e0), s9 = C(0x1e4), s10 = C(0x1e8), s12 = C(0x1f0), s13 = C(0x1f4), s14 = C(0x1f8);
	float s44 = C(0x30), s45 = C(0x34), s46 = C(0x38), s47 = C(0x3c);
	float s16 = C(0x00), s17 = C(0x04), s18 = C(0x08), s19 = C(0x0c);
	float s20 = C(0x10), s21 = C(0x14), s22 = C(0x18), s23 = C(0x1c);
	float s58 = C(0x1c0), s59 = C(0x1c4);
	float s24 = draw.o0.x, s25 = draw.o0.y, s26 = draw.o0.z;
	float s28 = draw.o1.x, s29 = draw.o1.y, s30 = draw.o1.z;
	float s48 = draw.o2.x, s49 = draw.o2.y, s50 = draw.o2.z;
	float s36 = draw.t3.x, s37 = draw.t3.y, s38 = draw.t3.z, s39 = draw.t3.w;
	const float k = 15000.0; // (0x466a6000)

	precise float v0 = position.x, v1 = position.y, v2 = position.z;
	precise float v3, v4, v5, v6, v7;
	v4 = s36 - s57;              // v_sub_f32_sdwa v4, s36, s57
	v6 = s34 * v2;               // v_mul_f32 v6, s34, v2
	v5 = s6 * v2;                // v_mul_f32 v5, s6, v2
	v7 = s2 * v2;                // v_mul_f32 v7, s2, v2
	v2 = v1 * s33 + v6;          // v_mad_f32 v2, v1, s33, v6
	v5 = s5 * v1 + v5;           // v_mac_f32 v5, s5, v1
	v7 = s1 * v1 + v7;           // v_mac_f32 v7, s1, v1
	v2 = s32 * v0 + v2;          // v_mac_f32 v2, s32, v0
	v1 = s4 * v0 + v5;           // v_mad_f32 v1, s4, v0, v5
	v7 = s0 * v0 + v7;           // v_mac_f32 v7, s0, v0
	v6 = s50 * v2;               // v_mul_f32 v6, s50, v2
	v5 = s30 * v2;               // v_mul_f32 v5, s30, v2
	v0 = s26 * v2;               // v_mul_f32 v0, s26, v2
	v2 = s49 * v1 + v6;          // v_mad_f32 v2, s49, v1, v6
	v5 = s29 * v1 + v5;          // v_mac_f32 v5, s29, v1
	v0 = s25 * v1 + v0;          // v_mac_f32 v0, s25, v1
	v2 = s48 * v7 + v2;          // v_mac_f32 v2, s48, v7
	v5 = s28 * v7 + v5;          // v_mac_f32 v5, s28, v7
	v0 = s24 * v7 + v0;          // v_mac_f32 v0, s24, v7
	v2 = k * v2 - vcc;           // v_mad_f32 v2, 0x466a6000, v2, -vcc_lo
	v1 = v5 * k + v4;            // v_madmk_f32 v1, v5, 0x466a6000, v4
	v0 = k * v0 - s56;           // v_mad_f32 v0, 0x466a6000, v0, -s56
	v2 = s39 * vcc + v2;         // v_mad_f32 v2, s39, vcc_lo, v2
	v1 = (-s38) * v4 + v1;       // v_mac_f32_e64 v1, -s38, v4
	v0 = s37 * s56 + v0;         // v_mad_f32 v0, s37, s56, v0
	v5 = s42 * v2;               // v_mul_f32 v5, s42, v2
	v4 = s14 * v2;               // v_mul_f32 v4, s14, v2
	v6 = s10 * v2;               // v_mul_f32 v6, s10, v2
	v2 = s41 * v1 + v5;          // v_mad_f32 v2, s41, v1, v5
	v7 = s13 * v1 + v4;          // v_mad_f32 v7, s13, v1, v4
	v6 = s9 * v1 + v6;           // v_mac_f32 v6, s9, v1
	v2 = s40 * v0 + v2;          // v_mac_f32 v2, s40, v0
	v7 = s12 * v0 + v7;          // v_mac_f32 v7, s12, v0
	v6 = s8 * v0 + v6;           // v_mac_f32 v6, s8, v0
	v5 = s46 * v2 + s47;         // v_mad_f32 v5, s46, v2, s47
	v4 = s18 * v2 + s19;         // v_mad_f32 v4, s18, v2, s19
	v2 = s22 * v2 + s23;         // v_mad_f32 v2, s22, v2, s23
	v1 = s45 * v7 + v5;          // v_mad_f32 v1, s45, v7, v5
	v4 = s17 * v7 + v4;          // v_mac_f32 v4, s17, v7
	v0 = s21 * v7 + v2;          // v_mad_f32 v0, s21, v7, v2
	v1 = s44 * v6 + v1;          // v_mac_f32 v1, s44, v6
	v2 = s16 * v6 + v4;          // v_mad_f32 v2, s16, v6, v4
	v0 = s20 * v6 + v0;          // v_mac_f32 v0, s20, v6
	v2 = s58 * v1 + v2;          // v_mac_f32 v2, s58, v1
	v0 = s59 * v1 + v0;          // v_mac_f32 v0, s59, v1
	v3 = 0.0;                    // v_mov_b32 v3, 0
	gl_Position = vec4(v2, v0, v3, v1); // exp pos0 v2, v0, v3, v1
}
