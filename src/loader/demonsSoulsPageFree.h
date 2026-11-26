#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// (m5, 11-25) KYTY_ENGINE_FREE: las paginas de los bloques que el asignador de paginas del juego (BPE
// PlatformPageAllocator, 01.007.000) deja libres y mapeados se devuelven a Windows cuando llevan ~30 s libres.
// En PS5 esa memoria es libre; aqui es RAM que el proceso retiene (0,6-2,8 GiB segun el mundo).
// Sin parches en el codigo del juego: un hilo del emulador toma el cerrojo del propio asignador cada 2 s, recorre sus
// listas de libres (solo lectura) y suelta (MEM_RESET + fuera del working set) trozos de 64 KiB, cada uno bajo el
// cerrojo tras comprobar que el bloque sigue libre. Si el juego reutiliza el bloque, Windows le da paginas al tocarlas
// (no hay que devolver nada a mano); se cuenta como "devuelto".
//   =giveback (o 1): activo. Por defecto con 16 GiB o menos (RAM fisica o limite del job), solo Windows.
//   =walk: solo recorre y cuenta.  =0: apagado.
// Linea cada 30 s: "[page-free] 30 s: ...".
namespace Loader {
struct Program;
namespace DemonsSoulsPageFree {
void Install(Program* program);
void Clear();

// Ghidra (eboot 01.007.000, base 0): la liberacion de un bloque es FUN_0080f550 (llamada desde 0x8019a1, 0x801a4b,
// 0x8c7e44). Cerrojo: cmp qword [0x2ee4b60],0 (+0x27, operando en +0x2a); estadisticas: mov rax,[0x2ee4d60] (+0x97).
inline constexpr std::string_view FreeBlock = "55 48 89 e5 41 57 41 56 53 48 83 ec 18 4c 8b 35 ?? ?? ?? ?? 48 89 fb 49 c7 c7 ff "
                                              "ff ff ff 31 c9 49 8b 06 48 89 45 e0 48 83 3d ?? ?? ?? ?? 00 74";
inline constexpr std::string_view FreeBlockBody =
    "48 8b 05 ?? ?? ?? ?? 48 8d 73 20 0f b6 4b 70 48 c7 43 20 00 00 00 00 4c 8b 4b 28 4c 8b 43 30 48 8d 14 89";
inline constexpr size_t FreeBlockBodyAt = 0x97;
inline constexpr size_t LockOperandAt = 0x2a, LockOperandEnd = 0x2f;
inline constexpr size_t StatsOperandAt = 0x9a, StatsOperandEnd = 0x9e;
// Descriptor de bloque: tamano +0x50, bytes mapeados +0x58, direccion +0x60; nodo de lista +0x20 (dueno en +0x20,
// siguiente +0x30, cabeza de su lista +0x38).
inline constexpr uint32_t SizeAt = 0x50, MappedAt = 0x58, AddressAt = 0x60, NodeAt = 0x20, NextAt = 0x30, ListAt = 0x38;
// Pools del objeto de estadisticas, 0xa0 entre si (lea rdx,[rcx+rcx*4]; shl edx,5): cabeza de libres +0x28 (primer
// nodo +0x38), bloques libres +0x48 (inc dword [rdi+0x48]), paginas de 64 KiB mapeadas libres +0x70.
inline constexpr uint64_t Pools = 2, PoolStride = 0xa0, FreeHeadAt = 0x28, FreeFirstAt = 0x38, FreeCountAt = 0x48,
                          FreePagesAt = 0x70;
} // namespace DemonsSoulsPageFree
} // namespace Loader
