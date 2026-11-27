#pragma once
// (vg1, solo mide; con KYTY_TECHO=1) Por que un draw acaba en el manejador PM4 (camino normal, no tabla): una linea
// [techo-normal] por ventana de 5 s con draws/fot por motivo y por paquete. Coste apagado: una carga relajada.
//   sin intento: el paquete no tiene via de tabla (DRAW_INDIRECT no indexado, DRAW_INDEX_INDIRECT con cabecera rara...)
//   decal-safe:  KYTY_DECAL_SAFE lo manda al camino normal a proposito (mezcla + prueba de profundidad sin escritura)
//   forma:       primitiva no lista de triangulos o indices de 8 bits (la tabla no los toma)
//   tabla:       la tabla lo rechazo (sin variante/estado/programa aun, pide store, objetivos cambiados...)
#include "local/techo.h"

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace NormalWhy {

enum Why : uint8_t { NoTry = 0, Decal, Shape, Table, Whys };
inline constexpr const char* kNames[Whys] = {"sin intento", "decal-safe", "forma", "tabla"};
inline thread_local uint8_t  t_why = NoTry;
inline std::atomic<uint64_t> g_n[Whys] {};
inline std::atomic<uint64_t> g_op[Whys][4] {}; // por paquete: DRAW_INDIRECT, DRAW_INDEX_INDIRECT(+MULTI), INDEX_2/OFFSET_2, AUTO
inline std::atomic<uint64_t> g_shape[32] {};   // (ti1) "forma" por tipo de primitiva (VGT_PRIMITIVE_TYPE; 32 = indices de 8 bits)

[[nodiscard]] inline bool On() { return Techo::On(); }

// En el manejador: opcode de draw, primitiva lista de triangulos?, tipo de indice.
inline void Count(uint32_t opcode, uint32_t prim, bool tri_list, uint32_t index_type) {
	if (!On()) return;
	uint8_t w = t_why;
	if (w == Table && (!tri_list || index_type > 1)) w = Shape;
	if (w == Shape) g_shape[!tri_list ? (prim & 31u) : 0u].fetch_add(1, std::memory_order_relaxed);
	const uint32_t op = opcode == 0x24 ? 0 : (opcode == 0x25 || opcode == 0x38) ? 1 : (opcode == 0x27 || opcode == 0x35) ? 2 : 3;
	g_n[w].fetch_add(1, std::memory_order_relaxed);
	g_op[w][op].fetch_add(1, std::memory_order_relaxed);
	t_why = NoTry;
}

inline void Report(double frames) {
	if (!On() || frames <= 0) return;
	static constexpr const char* ops[4] = {"DRAW_INDIRECT", "DRAW_INDEX_INDIRECT", "INDEX_2/OFFSET_2", "INDEX_AUTO"};
	std::printf("[techo-normal] draws al camino normal por motivo (draws/fot):");
	for (uint32_t w = 0; w < Whys; ++w) {
		const double n = static_cast<double>(g_n[w].exchange(0, std::memory_order_relaxed)) / frames;
		std::printf(" | %s %.1f (", kNames[w], n);
		for (uint32_t o = 0; o < 4; ++o)
			std::printf("%s%s %.1f", o ? ", " : "", ops[o], static_cast<double>(g_op[w][o].exchange(0, std::memory_order_relaxed)) / frames);
		std::printf(")");
	}
	std::printf(" | forma por primitiva (0x1 puntos, 0x2-0x3 lineas, 0x5 abanico, 0x6 tira tri [la tabla la intento], 0x7 rect, 0x11 rect antiguo; 0x0 = lista tri con indices de 8 bits):");
	for (uint32_t t = 0; t < 32; ++t)
		if (const auto c = g_shape[t].exchange(0, std::memory_order_relaxed); c != 0)
			std::printf(" 0x%x %.1f", t, static_cast<double>(c) / frames);
	std::printf("\n");
	std::fflush(stdout);
}

} // namespace NormalWhy
