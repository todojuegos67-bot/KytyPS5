#version 450
// (F5 pase 2, KYTY_NATIVO_SOMBRAS_SKIN; renderer/dsSombrasNativas.inc) El VS de profundidad de las sombras de las
// entradas con skinning de Demon's Souls (hash 0x3866090d8246e47e, ES @386610000, 95 dwords, volcado del log
// 20261125-053825-022; llvm-mc gfx1013) operacion por operacion. No es el estatico (0x7833ddbaf675c1c3,
// gpu_ds_sombras.vert): sin la cadena 0x6e0/0x700 -> *15000 -> 0x1e0/0x200 -> 0x0/0x30, sin tabla[3]; la posicion ya
// esta en el bufer de skinning (tabla[0]+0x10, format_xyz), la matriz del objeto son 3 filas por instancia leidas con
// buffer_load_dwordx4 idxen del V# tabla[2]+0x10 (indice = USER_DATA s10 & 0xffff + instance_id; la CPU las copia ya
// indexadas), luego la 4x4 de la camara en tabla[1]+0xc0 y el desplazamiento por w de tabla[1]+0x1c0. Exporta z real
// (v9), el estatico exporta z = 0. param0 (xy-0x2d0, z-0x2d8, uv) no influye en la profundidad sin PS: no se exporta.
// v_mad/v_mac de GFX10: multiplicacion redondeada y luego suma (no fusionadas): `precise`.
layout(location = 0) in vec3 position; // v14, v15, v16

layout(std140, set = 0, binding = 0) uniform Camera {
	vec4 c[114]; // (0x720 bytes: tabla[1] = la vista de la cascada)
} camera;

layout(push_constant) uniform Draw {
	vec4 r0; // fila 0 del objeto (offset 0):  v6..v9
	vec4 r1; // fila 1 (offset 16):            v0..v3
	vec4 r2; // fila 2 (offset 32):            v10..v13
	vec4 unused; // (t3: el VS de skinning no lee tabla[3])
} draw;

out gl_PerVertex {
	vec4 gl_Position;
};

float C(int byte_offset) {
	int i = byte_offset >> 2;
	return camera.c[i >> 2][i & 3];
}

void main() {
	// s_load_dwordx16 s[0:15], tabla[1], 0xc0; s_load_dwordx2 s[16:17], tabla[1], 0x1c0.
	float s0 = C(0xc0), s1 = C(0xc4), s2 = C(0xc8), s3 = C(0xcc);
	float s4 = C(0xd0), s5 = C(0xd4), s6 = C(0xd8), s7 = C(0xdc);
	float s8 = C(0xe0), s9 = C(0xe4), s10 = C(0xe8), s11 = C(0xec);
	float s12 = C(0xf0), s13 = C(0xf4), s14 = C(0xf8), s15 = C(0xfc);
	float s16 = C(0x1c0), s17 = C(0x1c4);

	precise float v14 = position.x, v15 = position.y, v16 = position.z;
	precise float v6 = draw.r0.x, v7 = draw.r0.y, v8 = draw.r0.z, v9 = draw.r0.w;   // buffer_load_dwordx4 v[6:9] (+0)
	precise float v0 = draw.r1.x, v1 = draw.r1.y, v2 = draw.r1.z, v3 = draw.r1.w;   // buffer_load_dwordx4 v[0:3] (+16)
	precise float v10 = draw.r2.x, v11 = draw.r2.y, v12 = draw.r2.z, v13 = draw.r2.w; // buffer_load_dwordx4 v[10:13] (+32)
	precise float v4;
	v13 = v16 * v12 + v13;       // v_mac_f32 v13, v16, v12
	v3  = v16 * v2 + v3;         // v_mac_f32 v3, v16, v2
	v9  = v16 * v8 + v9;         // v_mac_f32 v9, v16, v8
	v13 = v15 * v11 + v13;       // v_mac_f32 v13, v15, v11
	v11 = v1 * v15 + v3;         // v_mad_f32 v11, v1, v15, v3
	v9  = v15 * v7 + v9;         // v_mac_f32 v9, v15, v7
	v13 = v14 * v10 + v13;       // v_mac_f32 v13, v14, v10
	v11 = v14 * v0 + v11;        // v_mac_f32 v11, v14, v0
	v3  = v6 * v14 + v9;         // v_mad_f32 v3, v6, v14, v9
	v6  = s2 * v13 + s3;         // v_mad_f32 v6, s2, v13, s3
	v8  = s14 * v13 + s15;       // v_mad_f32 v8, s14, v13, s15
	v0  = s6 * v13 + s7;         // v_mad_f32 v0, s6, v13, s7
	v9  = s10 * v13 + s11;       // v_mad_f32 v9, s10, v13, s11
	v7  = s1 * v11 + v6;         // v_mad_f32 v7, s1, v11, v6
	v6  = s13 * v11 + v8;        // v_mad_f32 v6, s13, v11, v8
	v0  = s5 * v11 + v0;         // v_mac_f32 v0, s5, v11
	v9  = s9 * v11 + v9;         // v_mac_f32 v9, s9, v11
	v7  = s0 * v3 + v7;          // v_mac_f32 v7, s0, v3
	v6  = s12 * v3 + v6;         // v_mac_f32 v6, s12, v3
	v8  = s4 * v3 + v0;          // v_mad_f32 v8, s4, v3, v0
	v9  = s8 * v3 + v9;          // v_mac_f32 v9, s8, v3
	v4  = v6 * s16 + v7;         // v_mad_f32 v4, v6, s16, v7
	v8  = s17 * v6 + v8;         // v_mac_f32 v8, s17, v6
	gl_Position = vec4(v4, v8, v9, v6); // exp pos0 v4, v8, v9, v6 done
}
