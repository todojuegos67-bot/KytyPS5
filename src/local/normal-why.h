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

[[nodiscard]] inline bool On() { return Techo::On(); }

// En el manejador: opcode de draw, primitiva lista de triangulos?, tipo de indice.
inline void Count(uint32_t opcode, bool tri_list, uint32_t index_type) {
	if (!On()) return;
	uint8_t w = t_why;
	if (w == Table && (!tri_list || index_type > 1)) w = Shape;
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
	std::printf("\n");
	std::fflush(stdout);
}

} // namespace NormalWhy
