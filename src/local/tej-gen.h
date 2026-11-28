#pragma once
// (tej-huecos) KYTY_TEJ=3: el trabajador invalida su estado grafico al abrir cada paquete muestreado (tej-coser.h,
// paso A2) para que la lista capture todos los binds. Esa invalidacion sube GraphicsGeneration y el camino de la tabla
// volvia a BUSCAR el buffer de indices (same_index falso) en vez de reutilizarlo: en la sombra la busqueda falla
// ("buffer lookup", 37-260 huecos/fot en el log 20261128-115504-241) y el paquete queda con huecos. Aqui se recuerda
// el salto de generacion que hizo la sombra (de -> a): si es el unico cambio desde la ultima emision, se re-graba el
// MISMO buffer de indices (mismo handle y offset que ya estaba ligado; la captura lo ve) sin buscarlo.
#include <cstdint>

namespace TejGen {
inline thread_local uint64_t t_from = ~uint64_t {0}, t_to = ~uint64_t {0};
// Solo se llama en el trabajador, justo alrededor de su InvalidateGraphicsState.
inline void Note(uint64_t from, uint64_t to) noexcept { t_from = from, t_to = to; }
// La emision anterior era de la generacion `emitted` y ahora el buffer esta en `now`: solo la sombra la cambio.
[[nodiscard]] inline bool OnlyShadowBump(uint64_t emitted, uint64_t now) noexcept {
	return emitted == t_from && now == t_to && t_to != ~uint64_t {0};
}
} // namespace TejGen
