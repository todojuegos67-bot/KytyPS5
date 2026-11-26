#pragma once

// (p1, 11-25) KYTY_PORT_ALLOC=1 (apagado por defecto hasta probarlo): el asignador de paginas del juego
// (bpe::memory::PlatformPageAllocator, eboot 01.007.000) se sustituye por uno nativo de Windows.
//
// Ghidra (base 0): RequestVirtualPages_NoThreadSafety 0x80d9a0 (todas las altas, compromisos y descompromisos de paginas,
// llamada con el cerrojo [0x2ee4b60] ya tomado por 0x80d640 / 0x80e620) y la liberacion 0x80f550 (toma el cerrojo).
// Con ellas sustituidas, nada del juego vuelve a usar: el mapa de bits de paginas fisicas de 64 KiB del pool de 12 GiB
// (0x80d9a0 / 0x80eb90), la cola de sceKernelBatchMap (FlushMemoryRequestQueue 0x80f2c0), la reutilizacion de bloques
// liberados aun mapeados (0x80e7a0) ni el desmapeo con sceKernelVirtualQuery + ReserveVirtualRange(FIXED) (0x80eb90).
// Cada bloque: reserva de direcciones solo de lo pedido, compromiso al tocar (commit-real), devolucion real a Windows
// al liberar. Se rellenan los mismos descriptores (PageAllocationInfo), listas, tablas de dueno y estadisticas que lee
// el resto del juego (detalle en el .cpp). Con KYTY_PORT_ALLOC=1 el pool de PS5 del juego queda en 64 MiB
// (KYTY_MEM_DIRECTA_MB=64 si no se puso otro valor) y KYTY_ALLOC_NATIVO no se instala (este lo sustituye).
// Linea "[port-alloc]" cada 5 s.
namespace Loader {
struct Program;
namespace DemonsSoulsPortAlloc {
void Install(Program* program);
void Clear();
// Pedido por el entorno (KYTY_PORT_ALLOC=1); lo consulta DemonsSoulsAllocNativo para no enganchar la misma funcion.
bool Requested();
} // namespace DemonsSoulsPortAlloc
} // namespace Loader
