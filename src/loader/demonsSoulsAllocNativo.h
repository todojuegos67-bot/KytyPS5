#pragma once

// (m6, 11-25) KYTY_ALLOC_NATIVO (=1 por defecto; =0 apagado): la memoria que el asignador de paginas del juego
// (BPE PlatformPageAllocator, eboot 01.007.000) libera se devuelve a Windows en el acto (MEM_DECOMMIT: RAM y commit),
// en vez de quedarse comprometida en sus listas de libres.
// Ghidra (base 0): Init 0x80c200 reserva min(memoria directa, 12 GiB) como memoria fisica SIN mapear
// ("physicalMemoryToKeepUnmapped", mapa de bits de paginas de 64 KiB); cada peticion mapea solo lo que pide
// (0x80d9a0 -> sceKernelMapDirectMemory) y cada bloque liberado (0x80f550) entra mapeado en la lista libre de su pool
// (0 = CPU/onion, 1 = GPU/garlic; registro cada 0xa0 en [0x2ee4d60]) para reutilizarse sin llamada al sistema.
// Con commit-real (KYTY_COMMIT_REAL) la memoria ya se reserva al mapear y se compromete al tocar; aqui la liberacion
// pasa por un gancho en la entrada de 0x80f550 (sin cambiar su logica): antes de que el bloque entre en la lista libre
// (aun es del que lo libera: nadie mas puede tocarlo) se invalidan las caches de la GPU sobre el y se descompromete.
// Si el juego lo reutiliza, el primer toque lo vuelve a comprometer a ceros (el nuevo dueno escribe antes de leer).
// Paginas vigiladas (raiz, PageManager, offload) no se descomprometen: quedan para PageFree (red de seguridad).
// Linea "[alloc-nativo]" cada 5 s.
namespace Loader {
struct Program;
namespace DemonsSoulsAllocNativo {
void Install(Program* program);
void Clear();
} // namespace DemonsSoulsAllocNativo
} // namespace Loader
