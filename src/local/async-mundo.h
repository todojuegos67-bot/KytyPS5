#pragma once
// (ASYNC-MUNDO, 11-27) KYTY_ASYNC_POR_MUNDO=1 (por defecto 0, hasta la prueba del usuario): enableAsyncCompute=false solo
// en la Torre de Latria (m04, mundo 3), true en el resto. Medido (demons-souls.md): false da Latria +4,9 % y Shrine of
// Storms -4..-10 %, por eso no es un valor global.
// Ghidra 01.007.000: el juego NO copia el convar al arrancar; lo lee de la tabla de convars (tabla 0x902F13B40, byte
// +0xca0) cada vez que monta el grafo de render: 0x9f4435 `mov rax,[rbx+0x68]; cmp byte [rax+0xca0],1` decide la cola
// (0x91e640 esi=2 = async compute) y 0x9f44fa / 0x90f68f / 0x9f2f04 igual. Asi que cambiar el byte en caliente vale.
// Se cambia solo cuando cambia el mapa PEDIDO (*0x27bf378 +0xc, el mismo que usa vram-tope.h), es decir durante la
// carga, no a mitad de un mundo. Si el byte no es 0/1, o ya estaba en false al arrancar (lo puso el usuario), no se toca.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vram-juego.h"

namespace AsyncMundo {

constexpr uint64_t kTablaConvars = 0x902F13B40ull;
constexpr uint64_t kIdAsync      = 0xca0;
constexpr uint64_t kMapPtrRva    = 0x27bf378;
constexpr uint64_t kLectorRva    = 0x9f4435; // mov rax,[rbx+0x68]; cmp byte [rax+0xca0],1
constexpr int      kMapNone      = 0x63646464;
constexpr int      kLatria       = 4; // m04

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_ASYNC_POR_MUNDO");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

// Cada ~1 s desde el hilo de render.
inline void Tick() {
#if KYTY_VRAMJUEGO_OK
	static int  estado = 0; // 0 sin comprobar, 1 activo, -1 apagado
	static int  mundo  = -1;
	if (estado < 0 || !On()) return;
	auto* byte = reinterpret_cast<volatile uint8_t*>(kTablaConvars + kIdAsync);
	if (estado == 0) {
		static const uint8_t pat[] = {0x48, 0x8b, 0x43, 0x68, 0x80, 0xb8, 0xa0, 0x0c, 0x00, 0x00, 0x01};
		const auto*          at    = reinterpret_cast<const uint8_t*>(VramJuego::kEbootBase + kLectorRva);
		if (IsBadReadPtr(at, sizeof(pat)) != 0 || std::memcmp(at, pat, sizeof(pat)) != 0 ||
		    IsBadReadPtr(const_cast<uint8_t*>(byte), 1) != 0) {
			std::printf("[async-mundo] eboot no es 01.007.000: apagado\n");
			std::fflush(stdout);
			estado = -1;
			return;
		}
		if (*byte != 1) {
			std::printf("[async-mundo] enableAsyncCompute ya es %u al arrancar (lo fijo el usuario o no es la tabla): no se toca\n",
			            static_cast<unsigned>(*byte));
			std::fflush(stdout);
			estado = -1;
			return;
		}
		estado = 1;
	}
	auto* const* pp = reinterpret_cast<uint8_t* const*>(VramJuego::kEbootBase + kMapPtrRva);
	if (IsBadReadPtr(pp, 8) != 0 || *pp == nullptr || IsBadReadPtr(*pp + 0xc, 4) != 0) return;
	int id = 0;
	std::memcpy(&id, *pp + 0xc, 4);
	if (id == kMapNone || id <= 0 || static_cast<uint32_t>(id) >= 0x8000000u) return;
	const int m = id >> 24;
	if (m == mundo) return;
	mundo           = m;
	const uint8_t v = m == kLatria ? 0 : 1;
	if (*byte > 1) { estado = -1; return; }
	if (*byte != v) {
		*byte = v;
		std::printf("[async-mundo] mapa 0x%08x (m%02d): enableAsyncCompute=%s\n", static_cast<uint32_t>(id), m,
		            v != 0 ? "true" : "false (Latria)");
		std::fflush(stdout);
	}
#endif
}

} // namespace AsyncMundo
