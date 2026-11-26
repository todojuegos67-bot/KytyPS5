#pragma once
// (TEX-GUARDA, 11-26) Paginas "solo en la GPU" (TexNativa::AddOwned) vigiladas contra escrituras de la CPU.
//
// Crashes del 11-26 con KYTY_TEX_NATIVA=1 (eboot+0x8db03c; 21:37 CoherentMove con origen basura; 22:05 eboot+0x8e4f46,
// salto por una vtable basura): el juego reutiliza memoria de un rango "solo en la GPU" sin pasar por el gancho de free
// (0x802f90) ni por un unmap (op 2 de 0x7f9f70). La propiedad y la epoca (WB-EPOCA) no cambiaban, y la descarga
// VRAM -> RAM posterior (NativeRestoreBeforeDelete: borrado/aparcado, y AGUA) escribia bytes viejos de textura sobre
// objetos vivos del monton.
//
// Ahora toda pagina que toca un rango "solo en la GPU" queda protegida contra escritura (solo-lectura) mientras lo sea.
// Una escritura de la CPU del juego sobre ella hace fault y, ANTES de que la escritura siga: RemoveOwned de esa pagina
// (epoca nueva: ninguna descarga posterior la pisa; el juego vuelve a ser el dueno de la RAM), se devuelve la
// proteccion de escritura y la instruccion se repite. Coste: una VirtualProtect por trozo nuevo (AddOwned) y una por
// pagina escrita; los pixeles de textura no los escribe la CPU (el memcpy 0x901459 lo intercepta NativeChunkCopy, que
// no toca la RAM; lo que deja pasar a la RAM sale antes del registro).
//
// Convivencia con la vigilancia del emulador: si la pagina ya estaba sin escritura (solo-lectura / sin acceso del gestor
// de paginas) no se toca (la escritura hace fault igual: aqui se retira la propiedad y el fault sigue a su manejador).
// Cuando el emulador le da escritura (ProtectMappedUnlocked -> AfterProtect) se vuelve a quitar. Solo paginas
// comprometidas: una no comprometida no queda vigilada (Intacta = false: ninguna descarga la escribe).
#include <atomic>
#include <cstdint>

namespace TexGuarda {

// Ventana de la VirtualMemo de las texturas (TexNativa::kVmBegin / kVmEnd).
constexpr uint64_t kBegin = 0x1000000000ull;
constexpr uint64_t kEnd   = 0x1400000000ull;

// Paginas que tocan [b, e): vigiladas (las ya vigiladas no cambian). Bajo OwnedMutex.
void Guard(uint64_t b, uint64_t e);
// Paginas enteras [pb, pe) (alineadas): fuera de la vigilancia, con su proteccion de escritura devuelta. Bajo OwnedMutex.
void Unguard(uint64_t pb, uint64_t pe);
// Toda pagina que toca [b, e) vigilada (ninguna escritura de la CPU desde que se vigilo sin retirar la propiedad).
bool Intact(uint64_t b, uint64_t e);
// ProtectMappedUnlocked, despues de cambiar [vaddr, vaddr + size) a un modo (escritura / ejecucion).
void AfterProtect(uint64_t vaddr, uint64_t size, bool write, bool exec);
// Retira la propiedad "solo en la GPU" de [pb, pe) (TexNativa::RemoveOwned): la llama el fault de escritura.
using OnWrite = void (*)(uint64_t pb, uint64_t pe);
void SetOnWrite(OnWrite f);

inline std::atomic<int64_t>  g_pages {0};                 // paginas vigiladas ahora
inline std::atomic<uint64_t> c_writes {0};                // escrituras de la CPU del juego (propiedad retirada)
inline std::atomic<uint64_t> c_emulator {0};              // escrituras del emulador (descarga en curso)
inline std::atomic<uint64_t> c_no_commit {0};             // paginas no vigiladas (sin comprometer / proteccion rara)
inline std::atomic<uint64_t> c_reprotect {0};             // veces que el emulador les dio escritura y se quito otra vez
inline std::atomic<uint64_t> c_wb_unguarded {0}, c_wb_unguarded_bytes {0}; // descargas no hechas: rango no vigilado
// El hilo escribe la RAM de un rango "solo en la GPU" bajo OwnedMutex (NativeRestoreBeforeDelete): sus faults solo abren.
inline thread_local int t_emulator = 0;
struct EmulatorWrite {
	EmulatorWrite() { ++t_emulator; }
	~EmulatorWrite() { --t_emulator; }
};

} // namespace TexGuarda
