#pragma once
// (MALLA-NATIVA) encendido por defecto (KYTY_MALLA_NATIVA=0 lo apaga): las mallas del juego en un VkBuffer propio, desde el codigo
// del juego y sin vigilar ni proteger su memoria (el camino de BUF-NATIVO colgaba: 11-24 09:08).
//
// Ghidra/objdump 01.007.000:
//  - 0xaa06a0(malla, &bloque, &indices, ...) crea la malla: deja en malla+0x298 el bloque de vertices (0xaa1f70) y en
//    malla+0x2a0 el de indices (0xac26c0); los V# (flags 0x800) de sus 15 CBuffer (malla+8+i*0x28) apuntan DENTRO.
//  - 0xaa1d10(malla) la destruye (libera los dos bloques en diferido, 0x903460).
//  - Lectores de CPU del contenido de un CBuffer: 0x909940 (devuelve la base del V#: 8 llamadas) y 0x909970 (copia sus
//    datos a otro sitio: 18 llamadas, entre ellas 0xbd3bf0, que copia un bufer entero a un bloque nuevo). En partida
//    ~257 llamadas/s (linea [raiz] SETTLE del log 20261124-090734-005).
//
// Fase 1 (KYTY_MALLA_NATIVA=1): al volver de 0xaa06a0 se registran los dos bloques [bloque, fin de sus V#). La primera
// vez que la GPU pide un rango dentro (BufferCache::ObtainBuffer, solo lectura) se crea UN VkBuffer (VRAM visible por la
// CPU) con las paginas enteras del bloque, copiado de una vez; desde entonces esos V# se resuelven a el sin vigilancia,
// sin sincronizar y sin registrar nada en la cache de buferes. La RAM del juego no se toca. 0xaa1d10 lo destruye
// (diferido a que la GPU acabe).
//
// Fase 2 (ademas KYTY_MALLA_NATIVA_RAM=1): 0x909940/0x909970 enganchados: un bloque leido por ellos se marca "leido por
// CPU". Un bloque con VkBuffer, en VRAM, sin lecturas en 30 s y cuyos bytes siguen siendo los subidos (hash) suelta las
// paginas internas (64 KiB de margen) con MEM_RESET. Una lectura posterior por 0x909940/0x909970 lo restaura ANTES,
// copiando del VkBuffer (sincrono). Limite: un lector que no pase por esas dos funciones (puntero guardado >30 s o
// acceso al V# en linea) leeria bytes perdidos; por eso va aparte.
//
// Fase 1b (11-24, tirones del log 20261124-092720-218: 62 >50 ms en 5987 fotogramas, draw.vertex_buffers ~340 ms en
// cada uno): crear el VkBuffer DENTRO de ObtainBuffer (hilo de render) hacia vmaCreateBuffer con la VRAM al limite
// (used 6330-6387 / budget 6134 MiB en el log): fuera del presupuesto el constructor de Buffer vacia la reserva de
// imagenes y reintenta (streamBuffer.cpp), sincrono en mitad del draw. Ahora un hilo propio (Preparer) lee el bloque y
// reserva/sube su memoria en cuanto 0xaa06a0 lo registra; el hilo de render solo envuelve lo preparado (vkCreateBuffer +
// bind) y, si la memoria no es visible por la CPU, graba la copia desde staging en el prologo de subidas
// (UploadPrologue). Mientras no este listo: camino normal. Un bloque fallido no se reintenta nunca.
//
// Fase 1c (11-24, log 20261124-094546-995: 0 preparados, 14458 fallidos, 20.3 s del hilo; fotograma 16.2 ms frente a
// 14.4 sin MALLA-NATIVA): el Preparer preparaba TODOS los bloques al registrarlos en 0xaa06a0 (~46000 en la partida,
// la mitad destruidos sin dibujarse nunca), leyendolos enteros de una vez (TryReadBacking >64 KiB: candados del mapa,
// de datos y de escritura de la CPU, los mismos que el juego y el hilo de render). Ahora:
//  - solo se prepara un bloque cuando la GPU lo pide por primera vez (Lookup lo encola; ese uso sigue por el camino
//    normal): el juego ya lo lleno (la GPU lo dibuja) y solo se prepara lo que se usa;
//  - lectura en trozos de 64 KiB (camino rapido sin candado de TryReadBackingToHost) con una pausa entre trozos;
//  - hilo con prioridad baja, un bloque cada 2 ms como mucho y 32 MiB por segundo;
//  - presupuesto: si la VRAM del monton no tiene un 10 % libre tras el bloque, no se reserva (camino normal);
//  - un fallo, del motivo que sea, no se reintenta nunca; cada motivo se cuenta en la linea.
//
// Fase 1d (11-24, log 20261124-100008-962: "Fatal: abort" a los ~17 s en Boletaria). Pila resuelta con el .map:
// ExceptionFilter -> GpuResourceManager::HandleFault -> BufferCache::ReadMemory -> CopyGuestReadback ->
// WriteBackGpuOwned -> WriteBacking (noexcept) -> TryWriteBacking -> GuestBackingStore::DataLock -> std::mutex::lock
// lanza "resource deadlock would occur" -> terminate -> abort. Un hilo que YA tenia el candado de datos fallo de pagina
// y el manejador volvio a pedirlo. Es el Preparer: TryReadBackingToHost (kyty_local_backing_read_mode = 0, el camino
// rapido nunca se usa) va SIEMPRE por TryTransferBacking, que copia con DataLock tomado; un bloque con paginas de la GPU
// (protegidas) falla dentro. Ahora el Preparer lee el bloque como lo lee el juego: memcpy desde su direccion, sin
// ningun candado del emulador tomado (un fallo de pagina lo resuelve el manejador como para el juego). Y ninguna
// excepcion sale de los hilos propios (try/catch con aviso en el log).
//
// Fase 1f (11-24, log 20261124-105347-821: 8481 mallas, solo 994 bloques vivos, 329 pedidos, 136 preparados, 133 VkBuffer:
// ~2 % de cobertura). Ghidra (OJO: eboot.raw del contenedor esta desplazado +0xbdb0: VA 0xaa06a0 = raw 0xaac450,
// VA 0xaa1d10 = raw 0xaadac0):
//  - 0xaa06a0(malla, &bloque_v, &indices) rellena 14 CBuffer (bucle 0x28..0x2c8 de 0x30: malla+8+i*0x28, i<14) con
//    0x915210(slot, fmt, stride, n, 0x800, datos) -> slot+0x18 flags, +0x20 = n*stride, V# = datos. Despues: malla+0x298 =
//    *bloque_v, malla+0x2a0 = indices.ptr (dos reservas DISTINTAS del mismo monton). malla+0x238 NO es un CBuffer (objeto
//    de 0xb0 bytes, 0xacf190; 0xaa1d10 lo destruye con 0xacf160): el slot 15 leia basura.
//  - 0xaa1d10 libera +0x298 y +0x2a0 con 0x903460 y los pone a 0.
//  CAUSA: BlockEnd tomaba para CADA bloque el mayor base+bytes de TODOS los slots con base en [bloque, bloque+256 MiB):
//  el bloque de vertices se estiraba hasta el final de los indices (otra reserva, mas arriba en el monton) y al reves,
//  mas el slot 15 basura. Cada malla nueva registraba un rango gigante que en OnBuild BORRABA los bloques de las mallas
//  vecinas (8481 -> 994 vivos), y esos rangos gigantes (> MaxBlock o destruidos antes de leerse) se perdian en silencio
//  en PrepareOne (329 pedidos -> 136 preparados).
//  Ahora: 14 slots; cada slot cuenta solo para el bloque mas cercano por debajo de su base (vertices o indices), asi cada
//  bloque acaba en el final de SUS datos. Los descartes silenciosos se cuentan y la linea da la cobertura (usos de la GPU
//  servidos por un VkBuffer de malla / usos de solo lectura dentro del rango de mallas).
//
// Fase 1g (11-24, log 20261124-110909-771: cobertura 75.1 %, 1585 VkBuffer, 1 fallo de lectura). El 25 % restante no
// estaba clasificado. Ghidra (juego / archivo eboot.raw = juego + 0xbdb0): CBuffer::Create 0x909460 / 0x915210
// (slot, fmt, stride, n, flags, datos): flags 0x800 = datos externos (V# = datos, sin copia); sin 0x800 reserva con
// 0x805aa0 (montones 0x2eacac8/0x2eacab0 o 0x2f701a0 en el archivo) y copia los datos dentro. 489 llamadas, 63 con
// 0x800/0x804/0x805: 0xaa06a0 es solo una (las demas: 0x912e60, 0x914600, 0x914e10, 0x9f4330, 0x9f8380, 0xa1ed10,
// 0xa1f930, 0xa20ce0, 0xab6080 (V# temporal por dibujo en la pila), 0xab6540, 0xad8bc0, 0xadc560...). Sin datos no se
// registra ninguna: varias son por dibujo (dinamicas). Ahora (solo diagnostico, sin coste fuera de KYTY_MALLA_NATIVA=1):
//  - un gancho a la entrada de 0x909460 (mismos 13 bytes que 0xaa06a0) apunta V# base -> llamador/flags/bytes;
//  - cada uso no servido se cuenta por motivo (hueco, pasa del bloque, fallido, preparandose, primer uso, sin prologo) y
//    los huecos se suman por llamador de CBuffer::Create: linea "[malla-nativa] no servidos" + los 10 primeros.
//  - el fallo de lectura: el Preparer lee paginas enteras [page_lo, page_hi); si una pagina sin ningun byte del bloque
//    (cola redondeada) no se puede leer, va a ceros en vez de tirar el bloque entero.
//
// Fase 1k (11-25, log 20261125-061906-403: 69 % de usos nativos, huecos 6153699, RAM devuelta 0 con RAM apagada).
// Que son los huecos (datos: logs con KYTY_MALLA_NATIVA_DIAG=1 20261124-112316-638 y -113316-937; Ghidra 01.007.000):
// el rango [lo, hi) de mallas es todo el monton general 0x7f9cf0 (0xaa1f70 y 0xac26c0 reservan ahi los bloques), y los
// huecos son OTROS bufers de ese monton que la GPU lee, ninguno es una malla:
//  - 0xa92a94 (retorno 0xa92c77, 58 % de los huecos con origen): cada fotograma 0x7f9cf0(n*stride) + 0x909410 (suelta el
//    anterior) + CBuffer::Create(tipo 0x1d, 0x804, datos nuevos): bufer de instancias/huesos reescrito por la CPU por
//    fotograma (~33 KB).
//  - 0xa20750 (0xa20a2b), 0x912e60 (0x912fdf/0x912f6d), 0xadd363, 0xad8cab: CBuffer 0x800 en la PILA (slot temporal,
//    0x9093a0 lo inicia) con datos recien escritos por dibujo (0x160 bytes, constantes).
//  - 0xa1edf8: 0x804 sobre +0x10 de un objeto que la CPU rellena (0xa0 por registro).
//  - 0xa91a38/0xa91a67: flags 0x4 SIN datos (15 MiB / 1 MiB): los escribe la GPU (salidas de computo).
//  - el 71 % restante sin origen: V# escritos sin CBuffer::Create (0x909890, 350 llamadas: reserva en el monton por
//    fotograma 0x2f643f0 y escribe la base del V# directamente).
// Ninguno es estatico: un VkBuffer copiado una vez daria datos viejos. Pasarlos a nativo pide otra pieza (el monton por
// fotograma 0x2f643f0 / 0x7f9cf0 respaldado por un VkBuffer mapeado persistente, sin copia), que es memoria del juego
// (dueno: heap-nativo), no este archivo. Por eso el 100 % de MALLAS ya es nativo salvo arranque (primer uso/preparandose)
// y 1 bloque ilegible; el % de usos del rango no puede subir aqui.
// RAM: Ghidra: dentro del modulo de mallas (0xa90000-0xad0000) solo 0xaa1d10 lee +0x298/+0x2a0; el contenido se lee
// desde la CPU solo por 0x909940 (8 llamadores, todos usan el puntero al momento) y 0x909970 (18, copia), los dos
// enganchados (restauran antes). Las sombras (demonsSoulsNativoSombras.inc) leen la estructura CMesh, no el bloque.
// KYTY_MALLA_NATIVA_RAM (APAGADO por defecto desde la revision 11-25, ver RamOn; =1 lo enciende) y suelta las paginas enteras DENTRO del bloque
// (antes 64 KiB de margen + alineado a 64 KiB: con bloques de ~26 KB de media no se soltaba ninguno).
//
// Fase 1l (11-25): local/nativo-cubierto.h. Las paginas enteras de 4 KiB de [inicio, end) de cada bloque se marcan
// cubiertas (DMallas) cuando el bloque queda servido en VkBuffer (Lookup) y se descubren al destruirse (0xaa1d10 ->
// OnFree -> EraseLocked; tambien solape en OnBuild) o al tirar su VkBuffer por firma distinta. Asi la vigilancia del
// emulador deja de proteger las mallas que ya no lee.
//
// Fase 2b (11-25): KYTY_MALLA_NATIVA_RAM ENCENDIDO por defecto (=0 lo apaga), sin vigilancia ni fault propio.
// Ghidra 01.007.000 (VA del juego):
//  - 0xaa06a0 tiene 3 llamadas: 0xac08bc (0xac0410, el cargador) y 0xabfd03/0xabfe13 (0xabfba0/0xabfde0). Estas dos
//    NO tienen ninguna referencia en todo el binario (ni call/jmp/lea ni puntero de 8 bytes en memoria): codigo muerto.
//    Toda malla viene de 0xac0410.
//  - 0xac0410: 0xaa1f70 (bloque de vertices: 0x7f9cf0 reserva + 0x813130 lee del archivo + arreglo de punteros internos)
//    y 0xac26c0 (indices: 0x7f9cf0 + 0x813130) llenan los bloques ANTES de 0xaa06a0, que se los queda (pone a 0 el
//    origen: *param_2 = 0, param_3[2..3] = 0) y los guarda en +0x298/+0x2a0. Despues 0xac0410 no los toca.
//  - Unico destructor: 0xaa1d10 (llamado desde 0xac0410, 0xabf940, 0xac00f0 y los dos muertos; enganchado) libera con
//    0x903460 sin leer el contenido. En los logs con RAM encendida (065754, 071332, 072323, 074401, 075201) "borrados
//    por solape" = 0: ningun bloque se reutiliza sin pasar por 0xaa1d10.
//  - Lectores de CPU del contenido: 0x909940 (8 llamadores: 0xa25130, 0xa2bb00, 0xa2f910, 0xa34ab0, 0xa8f2f0, 0xa4fd60,
//    0xa6a8b0; los que escriben lo hacen en anillos de 2/3 CBuffer de OTROS objetos, indice &1/&3/%3, no en
//    malla+8+i*0x28) y 0x909970 (18, copia hacia fuera). Los dos enganchados (OnCpuRead toma ram_mutex la primera vez).
// Por tanto una malla no tiene escritor de CPU despues de 0xaa06a0. Datos: 5 sesiones con RAM=1, 5683 bloques
// soltados, "reescritos por la CPU" (hash antes de soltar) 0.
// Cambios:
//  - Solo se sueltan mallas que NUNCA leyo 0x909940/0x909970 y sin firma distinta (las demas se quedan en RAM).
//  - Ventana firma->drop: la comparacion y el MEM_RESET van bajo ram_mutex, el mismo candado por el que pasa la primera
//    lectura de CPU (OnCpuRead) y la destruccion (OnFree); y el bloque solo se suelta >=30 s despues de su primer uso
//    por la GPU (creacion completa y sin escritores: arriba).
//  - Restore comprueba pagina a pagina antes de copiar: pagina igual a la subida o a ceros (descartada) = intacta;
//    otra = la CPU escribio TRAS soltar: se cuenta ("reescritos por la CPU TRAS soltar", debe ser 0), esa pagina no se
//    pisa y el VkBuffer va a la fosa (camino normal para siempre).
//  - OnFree (0xaa1d10) ya no copia: la memoria se libera y nadie lee su contenido (antes 1273 restauraciones = 574 ms
//    en el hilo del juego, log 074401); solo comprueba.
//  RAM esperada: 18.5-38.3 MB devueltos (logs 075201 / 074401).
//
// Linea "[malla-nativa]" cada 30 s.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <exception>
#include <functional>
#include <map>
#include <initializer_list>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "buf-mem.h"         // (BUF-MEM: dedup y streaming de mallas)
#include "nativo-cubierto.h" // (fase 1l: paginas de mallas servidas en VkBuffer fuera de la vigilancia)

#if defined(_WIN32)
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread();
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void*, int);
#endif

namespace Libs::LibKernel::Memory {
bool     TryReadBackingToHost(uint64_t vaddr, void* data, uint64_t size);
uint64_t NativeOffloadDrop(uint64_t vaddr, uint64_t size, uint64_t* refused);
bool     NativeOffloadWrite(uint64_t vaddr, const void* data, uint64_t size);
} // namespace Libs::LibKernel::Memory

namespace MallaNativa {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_MALLA_NATIVA");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto; KYTY_MALLA_NATIVA=1 lo enciende
	}();
	return on;
}

// (fase 1h) Reserva de VRAM para mallas: dentro de ella (VRAM de mallas + bloque <= reserva) un bloque se crea mientras
// quepa bajo el presupuesto del driver (sin el 10 % de margen de las texturas). KYTY_MALLA_NATIVA_RESERVA_MB (256).
inline uint64_t ReserveBytes() {
	static const uint64_t b = [] {
		const char* v = std::getenv("KYTY_MALLA_NATIVA_RESERVA_MB");
		return (v != nullptr ? std::strtoull(v, nullptr, 10) : uint64_t {256}) << 20u;
	}();
	return b;
}
// (fase 1i) Diagnostico de la fase 1g (gancho a CBuffer::Create 0x909460, que corre en 489 llamadores y en cada dibujo
// desde 0xab6080, y el origen de los huecos en Lookup): solo con KYTY_MALLA_NATIVA_DIAG=1. Por defecto no se instala.
inline bool Diag() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_MALLA_NATIVA_DIAG");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
// (fase 1i) Firma: hash de los primeros 64 bytes del bloque al prepararlo; cada SigEvery usos servidos se compara con la
// memoria del juego. Distinta: la CPU reescribio la malla (el VkBuffer tiene datos viejos = parpadeo): se descarta y se
// vuelve a preparar; a la segunda, el bloque queda en el camino normal para siempre.
constexpr uint64_t SigBytes = 64;
constexpr uint32_t SigEvery = 64;
constexpr int RetrySeconds = 2; // (fase 1h: "sin presupuesto" se reintenta en el siguiente uso pasados 2 s)

inline bool RamOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_MALLA_NATIVA_RAM");
		// (ENCENDIDO por defecto desde la fase 2b, 11-25; KYTY_MALLA_NATIVA_RAM=0 lo apaga. Ver cabecera fase 2b: Ghidra
		// demuestra que ninguna malla se reescribe desde la CPU tras 0xaa06a0, y se comprueba en cada restauracion.)
		// (Historico, revision 11-25, apagado) Entre la firma Fnv(ReadGuest) y NativeOffloadDrop una escritura de la CPU se
		// pierde (MEM_RESET), y tras soltar las paginas quedan RW sin vigilancia: si la CPU escribe, la GPU sigue con el
		// VkBuffer viejo y Restore (0x909940/0x909970, OnFree) machaca los bytes nuevos. Cerrarlo exige poner las paginas
		// solo-lectura y atender su fault de escritura en el filtro de excepciones del kernel (fuera de este archivo, y
		// chocaria con las protecciones del gestor de paginas). Solo ahorraba ~18.5 MB: KYTY_MALLA_NATIVA_RAM=1 lo enciende.
		return On() && (v != nullptr && v[0] == '1'); // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

constexpr uint64_t Page       = 4096;
constexpr uint64_t Margin     = uint64_t {64} << 10u;
constexpr uint64_t MaxBlock   = uint64_t {256} << 20u;
constexpr uint64_t SlotFirst = 0x8, SlotStride = 0x28, SlotCount = 14; // (Ghidra: 14; +0x238 no es CBuffer)
constexpr uint64_t VertexBlock = 0x298, IndexBlock = 0x2a0;
constexpr uint32_t ExternFlag = 0x800u, AliasFlag = 0x400u;
constexpr int      IdleSeconds = 30;
constexpr uint64_t ReadChunk   = 0x10000;              // (el camino rapido de TryReadBackingToHost)
constexpr uint64_t RateBytes   = uint64_t {32} << 20u; // (por segundo, el hilo Preparer)

// Motivos de fallo de una preparacion (contados en la linea).
enum Fail : int { FailNone = 0, FailRead, FailProbe, FailBudget, FailAlloc, FailStaging, FailCount };

inline uint64_t ReadU64(uint64_t a) { return *reinterpret_cast<const uint64_t*>(a); }
inline uint64_t VsharpBase(uint64_t p) {
	return uint64_t {*reinterpret_cast<const uint32_t*>(p)} | (uint64_t {*reinterpret_cast<const uint16_t*>(p + 4)} << 32u);
}

// El final de los datos de un bloque: el mayor base+bytes de los CBuffer 0x800 propios cuyo V# cae dentro. `other` es
// el otro bloque de la malla: un V# por encima de el es suyo, no de este (fase 1f).
inline uint64_t BlockEnd(uint64_t mesh, uint64_t block, uint64_t other) {
	uint64_t end = 0;
	for (uint64_t i = 0; i < SlotCount; i++) {
		const uint64_t slot  = mesh + SlotFirst + i * SlotStride;
		const uint32_t flags = *reinterpret_cast<const uint32_t*>(slot + 0x18);
		const uint32_t size  = *reinterpret_cast<const uint32_t*>(slot + 0x20);
		if ((flags & ExternFlag) == 0 || (flags & AliasFlag) != 0 || size == 0) continue;
		const uint64_t base = VsharpBase(slot);
		if (base < block || base - block >= MaxBlock) continue;
		if (other > block && base >= other) continue; // (datos del otro bloque)
		end = std::max(end, base + size);
	}
	return end;
}

// (Preparer / Tick) Copia [vaddr, vaddr + size) como la lee el juego: desde su direccion, sin candados del emulador (con
// DataLock tomado, un fallo de pagina en la copia hacia que el manejador lo pidiera otra vez: abort).
//
// Fase 1e (11-24, log 20261124-101231-938: "Unhandled host exception" c0000005 en 0x375ac0000). Registros del blackbox:
// pc en vcruntime140 memcpy (rep movsb; r10 = base 0x7ff900bf0000), rsi = origen 0x375ac0000, rcx = 0x10000 (un trozo),
// rbx = page_lo 0x373800000, r12 = off 0x22c0000; retorno 0x141268540 = lambda de MallaNativa::StartPreparer (.map):
// el hilo Preparer LEYENDO un trozo. La pagina estaba "host reserved" (MEM_RESERVE, region 0x375ac0000+0x40000):
// descomprometida por el juego (la malla se destruye en 0xaa1d10 y sus bloques se liberan mientras el Preparer lee:
// OnFree no espera al Preparer) o fuera de lo comprometido del bloque. Para el emulador no es una pagina vigilada y
// el hilo no es del juego: el manejador no la resuelve y cierra el proceso.
// Ahora ReadGuest no toca la memoria con instrucciones de la CPU: ReadProcessMemory sobre el propio proceso (copia
// del nucleo). Una pagina no comprometida, sin acceso (RAIZ) o con guarda devuelve FALSE (ERROR_PARTIAL_COPY), sin
// excepcion, sin pasar por el manejador y sin candados del emulador: el bloque queda "failed" (camino normal). No hay
// ventana entre comprobar y copiar: la comprobacion ES la copia. Ademas el Preparer deja de leer en cuanto el bloque
// cambia de generacion (malla destruida) entre trozos.
#if defined(_WIN32)
extern "C" __declspec(dllimport) int __stdcall ReadProcessMemory(void*, const void*, void*, size_t, size_t*);
#endif
inline bool ReadGuest(uint64_t vaddr, void* data, uint64_t size) {
	if (vaddr == 0 || data == nullptr || size == 0) return false;
#if defined(_WIN32)
	size_t done = 0;
	return ReadProcessMemory(reinterpret_cast<void*>(static_cast<intptr_t>(-1)), reinterpret_cast<const void*>(vaddr), data,
	                         static_cast<size_t>(size), &done) != 0 &&
	       done == size;
#else
	std::memcpy(data, reinterpret_cast<const void*>(vaddr), size);
	return true;
#endif
}

inline uint64_t Fnv(const uint8_t* p, uint64_t n, uint64_t h = 0xcbf29ce484222325ull) {
	for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
	return h;
}
// (fase 1j) Firma de tres ventanas del bloque [0, len): primeros, del medio y ultimos SigBytes. `read(off, dst, n)`.
template <typename Read>
inline bool SigOf(uint64_t len, uint64_t* out, Read&& read) {
	uint8_t        w[SigBytes];
	const uint64_t n = std::min<uint64_t>(SigBytes, len);
	uint64_t       h = 0xcbf29ce484222325ull;
	for (const uint64_t off: {uint64_t {0}, (len - n) / 2 & ~uint64_t {3}, len - n}) {
		if (!read(off, w, n)) return false;
		h = Fnv(w, n, h);
	}
	*out = h;
	return true;
}

struct Block {
	uint64_t end = 0, owner = 0, gen = 0;
	void*    gpu = nullptr;        // (Libs::Graphics::Buffer*, del hilo de render)
	void*    pending = nullptr;    // (memoria preparada por el hilo Preparer, aun sin Buffer; opaco)
	uint8_t* mapped = nullptr;     // (su memoria visible por la CPU, paginas enteras desde page_lo)
	uint64_t page_lo = 0, page_hi = 0;
	uint64_t hash = 0;             // (de [inicio, end) al subirlo)
	bool     queued = false;       // (pedido por la GPU y en la cola del Preparer)
	bool     creating = false, failed = false, vram = false, read_by_cpu = false, modified = false;
	uint64_t reset_lo = 0, reset_hi = 0; // (paginas soltadas con MEM_RESET; 0: ninguna)
	std::chrono::steady_clock::time_point last_read;
	std::chrono::steady_clock::time_point retry_at {}; // (fase 1h: sin presupuesto; no antes de esto)
	uint32_t budget_fails = 0;
	uint64_t sig = 0;              // (fase 1i: hash de los primeros SigBytes al preparar)
	uint32_t uses = 0, stale = 0;  // (fase 1i: usos servidos desde la ultima comprobacion; firmas distintas)
	uint64_t dkey = 0;             // (BUF-MEM) clave de contenido: hash de TODAS sus paginas y su tamano (0: sin calcular)
	uint32_t seen_uses = 0, idle_s = 0; // (BUF-MEM) segundos seguidos sin usos de la GPU (streaming fuera de VRAM)
};

// (BUF-MEM) Deduplicacion: bloques con las mismas paginas (mismo hash y tamano) comparten un VkBuffer, con cuenta de
// referencias; solo la ultima referencia lo manda a la fosa.
struct Shared {
	uint32_t refs = 0;
	uint64_t key = 0, bytes = 0;
	uint8_t* mapped = nullptr;
	bool     vram = false;
};

struct Registry {
	std::mutex                                           ram_mutex; // (soltar / restaurar; antes que mutex)
	std::shared_mutex                                    mutex;
	std::map<uint64_t, Block>                            blocks; // inicio -> bloque
	std::unordered_map<uint64_t, std::vector<uint64_t>> by_owner;
	std::vector<void*>                                   graveyard; // (VkBuffer de mallas destruidas: hilo de render)
	std::unordered_map<uint64_t, void*>                  dedup_by_key; // (BUF-MEM) contenido -> VkBuffer vivo
	std::unordered_map<void*, Shared>                    dedup_refs;   // (BUF-MEM) VkBuffer -> referencias
	std::atomic<uint64_t>                                count {0}, lo {UINT64_MAX}, hi {0}, gen {0};
	// (MALLA-RAPIDA, 11-25) Sube (con mutex exclusivo) cada vez que un VkBuffer servido deja de valer o cambia el mapa de
	// bloques: invalida de golpe la cache de aciertos del hilo de render (Lookup), sin recorrer nada.
	std::atomic<uint64_t>                                epoch {1};
	// (MALLA-RAPIDA2, 11-25) Sube al dar de alta un bloque: solo invalida los HUECOS cacheados (un alta no quita validez a
	// ningun acierto: si solapa, EraseLocked ya mueve `epoch`).
	std::atomic<uint64_t>                                ins_epoch {1};
	std::atomic<uint64_t>                                fast_hits {0}, fast_gaps {0};
	std::atomic<bool>                                    hooked {false}, read_hooked {false};
	// Contadores.
	std::atomic<uint64_t> meshes {0}, freed {0}, created {0}, created_bytes {0}, vram_bytes {0}, sys_bytes {0}, failed {0};
	std::atomic<uint64_t> served {0}, cpu_reads {0}, reset_bytes {0}, resets {0}, restores {0}, restore_us {0};
	std::atomic<uint64_t> modified {0}, reset_refused {0};
	std::atomic<uint64_t> rewritten_after {0}, rewritten_pages {0}; // (fase 2b)
	// (fase 1f) Cobertura y descartes antes silenciosos.
	std::atomic<uint64_t> lookups {0}, overlapped {0}, dropped_gone {0}, dropped_big {0};
	std::atomic<bool>     worker {false}, dead {false};
	// Hilo Preparer: bloques registrados por 0xaa06a0 pendientes de preparar.
	std::mutex                                  queue_mutex;
	std::condition_variable                     queue_cv;
	std::deque<std::pair<uint64_t, uint64_t>>   queue; // (inicio, gen)
	// (puestos por BufferCache) prepare: lee nada, reserva y copia `bytes` a memoria de GPU (opaco o nullptr);
	// discard: libera lo preparado y nunca usado por la GPU (cualquier hilo).
	std::function<void*(uint64_t lo, uint64_t bytes, const uint8_t* data, int* reason)> prepare;
	std::function<void(void*)>                                             discard;
	std::atomic<bool>                                                      backend {false};
	std::atomic<uint64_t> prepared {0}, prepare_us {0}, staged {0}, requested {0};
	std::atomic<uint64_t> fail_reason[FailCount] {};
	// (fase 1g) Los usos del rango de mallas NO servidos, por motivo, y el origen de los que caen en un hueco.
	std::atomic<uint64_t> miss_gap {0}, miss_over {0}, miss_failed {0}, miss_preparing {0}, miss_first {0}, miss_prologue {0};
	std::atomic<uint64_t> miss_gap_known {0}, read_padded {0}, read_fail_addr {0};
	std::atomic<uint64_t> eboot {0};
	// (fase 1h) Reintentos tras "sin presupuesto", usos en espera de reintento, bytes pedidos al recolector de texturas
	// (VRAM-DESG los suma a lo que falta y expulsa texturas viejas), creados dentro de la reserva.
	std::atomic<uint64_t> retries {0}, miss_budget {0}, budget_want {0}, budget_want_total {0}, in_reserve {0};
	std::atomic<uint64_t> sig_checks {0}, sig_stale {0}, sig_off {0}; // (fase 1i)
	std::atomic<uint64_t> served_past_end {0}, spec_skipped {0}, grave_kept {0}; // (fase 1j)
	// (base del eboot: los llamadores se dan como eboot+X)
};

// (fase 1g, diagnostico) Cada CBuffer::Create (0x909460) deja aqui su V# base -> (llamador, flags, bytes). Tabla fija sin
// candados (carreras benignas: solo atribuye). Los usos que caen en un hueco del rango de mallas se suman por llamador.
constexpr uint64_t OriginSlots = uint64_t {1} << 16u, CallerSlots = 128;
struct OriginEntry {
	std::atomic<uint64_t> base {0}, caller {0}, flags_bytes {0};
};
struct CallerEntry {
	std::atomic<uint64_t> caller {0}, flags {0}, uses {0}, bytes {0};
};
inline OriginEntry* Origins() {
	static auto* t = new OriginEntry[OriginSlots];
	return t;
}
inline CallerEntry* Callers() {
	static auto* t = new CallerEntry[CallerSlots];
	return t;
}
inline uint64_t OriginHash(uint64_t base) { return ((base >> 4u) * 0x9e3779b97f4a7c15ull) >> 48u; }

inline Registry& Get() {
	static auto* r = new Registry(); // (nunca se destruye)
	return *r;
}

// (BUF-MEM, con mutex exclusivo) Suelta la referencia del bloque a su VkBuffer y descuenta sus bytes: a la fosa solo con
// la ultima referencia (dedup). Deja b.gpu sin tocar (lo pone a nullptr quien llama si hace falta).
inline void Bury(Registry& r, Block& b) {
	const uint64_t bytes = b.page_hi - b.page_lo;
	auto           sh    = r.dedup_refs.find(b.gpu);
	if (sh != r.dedup_refs.end() && sh->second.refs > 1) {
		sh->second.refs--;
		BufMem::g_dedup_bytes.fetch_sub(std::min<uint64_t>(bytes, BufMem::g_dedup_bytes.load()), std::memory_order_relaxed);
		BufMem::g_dedup_n.fetch_sub(BufMem::g_dedup_n.load() != 0 ? 1 : 0, std::memory_order_relaxed);
		return;
	}
	if (sh != r.dedup_refs.end()) {
		auto k = r.dedup_by_key.find(sh->second.key);
		if (k != r.dedup_by_key.end() && k->second == b.gpu) r.dedup_by_key.erase(k);
		r.dedup_refs.erase(sh);
	}
	r.graveyard.push_back(b.gpu);
	r.dead.store(true, std::memory_order_release);
	(b.vram ? r.vram_bytes : r.sys_bytes).fetch_sub(bytes, std::memory_order_relaxed);
}

// (ram_mutex + mutex) Quita un bloque: su VkBuffer a la fosa (lo destruye el hilo de render tras la GPU).
inline void EraseLocked(Registry& r, std::map<uint64_t, Block>::iterator it) {
	auto& b = it->second;
	if (b.pending != nullptr) {
		if (r.discard) r.discard(b.pending); // (nunca usado por la GPU)
		b.pending = nullptr;
	}
	if (b.gpu != nullptr) {
		NativoCubierto::Descubrir(it->first, b.end - it->first, NativoCubierto::DMallas); // (fase 1l: deja de servirse)
		Bury(r, b); // (BUF-MEM: dedup)
	}
	if (b.reset_hi != 0) r.reset_bytes.fetch_sub(b.reset_hi - b.reset_lo, std::memory_order_relaxed);
	auto o = r.by_owner.find(b.owner);
	if (o != r.by_owner.end()) {
		std::erase(o->second, it->first);
		if (o->second.empty()) r.by_owner.erase(o);
	}
	const bool had_gpu = b.gpu != nullptr;
	r.blocks.erase(it);
	r.count.store(r.blocks.size(), std::memory_order_relaxed);
	// (MALLA-TIRON, 11-26) Log 20261126-122830-166: 31992 mallas destruidas y solo 3569 con VkBuffer; cada baja movia
	// `epoch` y vaciaba la cache del hilo de render (2.98 M usos + 3.6 M huecos por shared_mutex + mapa, compitiendo con
	// el candado exclusivo de OnBuild/OnFree del hilo del juego: tiron1 carga/juego 17->29, otros 29->48). Un acierto
	// cacheado solo apunta a bloques con VkBuffer, y una baja solo agranda huecos (los huecos cacheados siguen valiendo):
	// sin VkBuffer no hay nada que invalidar.
	if (had_gpu) r.epoch.fetch_add(1, std::memory_order_release); // (MALLA-RAPIDA)
}

// Restaura las paginas soltadas de un bloque desde su VkBuffer. Con ram_mutex tomado y SIN mutex (la escritura pasa por
// una ventana del emulador que puede esperar al hilo de render; solo ram_mutex inserta/borra bloques o los suelta).
//
// (fase 2b) Antes, pagina a pagina: igual a la subida (aun en RAM) o toda a ceros (descartada por MEM_RESET) = intacta;
// otra cosa = la CPU la escribio DESPUES de soltar (contador "reescritos tras soltar": debe ser 0, Ghidra: nadie escribe
// una malla tras 0xaa06a0). Esa pagina NO se pisa (sus bytes son los del juego) y el VkBuffer va a la fosa (la GPU
// vuelve al camino normal con los datos actuales, para siempre). `write_back` = false (OnFree, 0xaa1d10: la memoria se
// libera con 0x903460 y nadie lee su contenido): solo se comprueba, sin copiar.
inline void Restore(Registry& r, uint64_t start, Block& b, bool write_back = true) {
	if (b.reset_hi == 0 || b.mapped == nullptr) return;
	const auto t0        = std::chrono::steady_clock::now();
	uint64_t   rewritten = 0, run = 0; // (run: inicio del tramo intacto pendiente de copiar; 0 = ninguno)
	auto       flush     = [&](uint64_t upto) {
        if (run != 0 && write_back)
            (void)Libs::LibKernel::Memory::NativeOffloadWrite(run, b.mapped + (run - b.page_lo), upto - run);
        run = 0;
	};
	{
		uint8_t page[Page];
		for (uint64_t at = b.reset_lo; at < b.reset_hi; at += Page) {
			const uint8_t* up  = b.mapped + (at - b.page_lo);
			bool           cpu = false;
			if (ReadGuest(at, page, Page) && std::memcmp(page, up, Page) != 0)
				cpu = std::any_of(page, page + Page, [](uint8_t c) { return c != 0; });
			if (cpu) {
				flush(at);
				rewritten++;
				continue; // (bytes del juego: no se pisan)
			}
			if (run == 0) run = at;
		}
		flush(b.reset_hi);
	}
	std::unique_lock lock(r.mutex);
	if (rewritten != 0) {
		r.rewritten_after.fetch_add(1, std::memory_order_relaxed);
		r.rewritten_pages.fetch_add(rewritten, std::memory_order_relaxed);
		b.modified = true;
		if (b.gpu != nullptr) { // (el VkBuffer ya no es la verdad: fuera)
			NativoCubierto::Descubrir(start, b.end - start, NativoCubierto::DMallas);
			Bury(r, b); // (BUF-MEM: dedup)
			b.gpu = nullptr, b.mapped = nullptr, b.queued = false, b.failed = true;
			r.epoch.fetch_add(1, std::memory_order_release); // (MALLA-RAPIDA)
		}
	}
	r.reset_bytes.fetch_sub(b.reset_hi - b.reset_lo, std::memory_order_relaxed);
	b.reset_lo = b.reset_hi = 0;
	r.restores.fetch_add(1, std::memory_order_relaxed);
	r.restore_us.fetch_add(static_cast<uint64_t>(
	                           std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
	                       std::memory_order_relaxed);
}

inline void StartWorker();

// (hilo del juego, al volver de CBuffer::Create 0x909460) Apunta el origen del V# (diagnostico, fase 1g).
inline void OnCBufferCreate(uint64_t slot, uint64_t return_address) {
	if (slot == 0) return;
	const uint64_t base = VsharpBase(slot);
	if (base == 0) return;
	const uint64_t flags = *reinterpret_cast<const uint32_t*>(slot + 0x18);
	const uint64_t bytes = *reinterpret_cast<const uint32_t*>(slot + 0x20);
	const uint64_t eb    = Get().eboot.load(std::memory_order_relaxed);
	auto&          e     = Origins()[OriginHash(base)];
	e.base.store(base, std::memory_order_relaxed);
	e.caller.store(return_address >= eb ? return_address - eb : 0, std::memory_order_relaxed);
	e.flags_bytes.store((flags << 32u) | bytes, std::memory_order_relaxed);
}

// (hilo de render) Un uso del rango de mallas que cae en un hueco: se suma a su llamador (si se conoce).
inline void CountGap(uint64_t vaddr) {
	auto&          r = Get();
	auto&          e = Origins()[OriginHash(vaddr)];
	if (e.base.load(std::memory_order_relaxed) != vaddr) return;
	r.miss_gap_known.fetch_add(1, std::memory_order_relaxed);
	const uint64_t caller = e.caller.load(std::memory_order_relaxed), fb = e.flags_bytes.load(std::memory_order_relaxed);
	auto*          t = Callers();
	for (uint64_t i = 0, h = (caller * 0x9e3779b97f4a7c15ull) >> 57u; i < CallerSlots; i++, h = (h + 1) % CallerSlots) {
		uint64_t c = t[h].caller.load(std::memory_order_relaxed);
		if (c == 0 && t[h].caller.compare_exchange_strong(c, caller + 1)) c = caller + 1;
		if (c != caller + 1) continue;
		t[h].flags.store(fb >> 32u, std::memory_order_relaxed);
		t[h].uses.fetch_add(1, std::memory_order_relaxed);
		t[h].bytes.fetch_add(fb & 0xffffffffu, std::memory_order_relaxed);
		return;
	}
}

// (hilo del juego, al volver de 0xaa06a0) Registra los dos bloques de la malla.
inline void OnBuild(uint64_t mesh) {
	if (mesh == 0) return;
	auto& r = Get();
	r.meshes.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard ram(r.ram_mutex);
	const uint64_t vblock = ReadU64(mesh + VertexBlock), iblock = ReadU64(mesh + IndexBlock);
	for (const uint64_t off: {VertexBlock, IndexBlock}) {
		const uint64_t block = off == VertexBlock ? vblock : iblock;
		if (block == 0) continue;
		const uint64_t end = BlockEnd(mesh, block, off == VertexBlock ? iblock : vblock);
		if (end <= block) continue;
		// (Otro bloque sobre el mismo sitio: memoria ya liberada y reutilizada; fuera.)
		auto it = r.blocks.upper_bound(block);
		if (it != r.blocks.begin() && std::prev(it)->second.end > block) it = std::prev(it);
		// (Sin restaurar: restaurar aqui pisaria los bytes nuevos. Ghidra/logs: no pasa, solape 0.)
		std::unique_lock lock(r.mutex);
		while (it != r.blocks.end() && it->first < end) {
			auto next = std::next(it);
			r.overlapped.fetch_add(1, std::memory_order_relaxed);
			EraseLocked(r, it);
			it = next;
		}
		Block b;
		b.end       = end;
		b.owner     = mesh;
		b.gen       = r.gen.fetch_add(1, std::memory_order_relaxed) + 1;
		b.page_lo   = block & ~(Page - 1);
		b.page_hi   = (end + Page - 1) & ~(Page - 1);
		b.last_read = std::chrono::steady_clock::now();
		r.blocks.emplace(block, b); // (se prepara cuando la GPU lo pida: Lookup)
		r.ins_epoch.fetch_add(1, std::memory_order_release); // (MALLA-RAPIDA2: el alta solo invalida huecos cacheados)
		r.by_owner[mesh].push_back(block);
		r.count.store(r.blocks.size(), std::memory_order_relaxed);
		if (block < r.lo.load(std::memory_order_relaxed)) r.lo.store(block, std::memory_order_relaxed);
		if (end > r.hi.load(std::memory_order_relaxed)) r.hi.store(end, std::memory_order_relaxed);
	}
	StartWorker();
}

// (hilo del juego, 0xaa1d10 ANTES del destructor) Sus bloques: si estaban soltados solo se comprueban (fase 2b: 0xaa1d10
// no lee el contenido y 0x903460 lo libera; el monton solo escribe sus cabeceras) y fuera; su VkBuffer a la fosa.
inline void OnFree(uint64_t mesh) {
	auto&           r = Get();
	std::lock_guard ram(r.ram_mutex);
	auto            o = r.by_owner.find(mesh);
	if (o == r.by_owner.end()) return;
	const auto starts = o->second;
	for (const auto s: starts) {
		auto it = r.blocks.find(s);
		if (it != r.blocks.end()) Restore(r, it->first, it->second, false); // (fase 2b: solo comprobar; se libera)
	}
	std::unique_lock lock(r.mutex);
	for (const auto s: starts) {
		auto it = r.blocks.find(s);
		if (it != r.blocks.end()) EraseLocked(r, it);
	}
	r.freed.fetch_add(1, std::memory_order_relaxed);
}

// (hilo del juego, ganchos de 0x909940/0x909970) Una lectura de CPU del bufer con V# en `vsharp`.
inline void OnCpuRead(uint64_t base) {
	auto& r = Get();
	if (r.count.load(std::memory_order_relaxed) == 0 || base < r.lo.load(std::memory_order_relaxed) ||
	    base >= r.hi.load(std::memory_order_relaxed))
		return;
	bool needs_restore = false;
	{
		std::shared_lock lock(r.mutex);
		auto             it = r.blocks.upper_bound(base);
		if (it == r.blocks.begin()) return;
		--it;
		if (base >= it->second.end) return;
		needs_restore = it->second.reset_hi != 0;
		if (!needs_restore && it->second.read_by_cpu) {
			// (sin cerrojo exclusivo: solo la hora, de este hilo; carrera benigna)
			it->second.last_read = std::chrono::steady_clock::now();
			return;
		}
	}
	std::lock_guard ram(r.ram_mutex); // (los bloques no cambian mientras)
	auto            it = r.blocks.upper_bound(base);
	if (it == r.blocks.begin()) return;
	--it;
	if (base >= it->second.end) return;
	auto& b = it->second;
	Restore(r, it->first, b); // (antes de que el juego lea)
	std::unique_lock lock(r.mutex);
	if (!b.read_by_cpu) r.cpu_reads.fetch_add(1, std::memory_order_relaxed);
	b.read_by_cpu = true;
	b.last_read   = std::chrono::steady_clock::now();
}

// (hilo de render) El bloque que contiene [vaddr, vaddr + size), con su VkBuffer, o nullptr (camino normal). Nunca crea
// ni sube nada aqui: `finish(pending, plo, bytes, &mapped, &vram)` solo envuelve lo que el hilo Preparer dejo listo
// (devuelve nullptr si aun no puede: se reintenta en el siguiente uso, sin coste).
template <typename Finish>
inline void* Lookup(uint64_t vaddr, uint64_t size, uint64_t* page_lo, Finish&& finish) {
	auto& r = Get();
	if (r.count.load(std::memory_order_relaxed) == 0 || vaddr < r.lo.load(std::memory_order_relaxed) ||
	    vaddr >= r.hi.load(std::memory_order_relaxed) || size == 0)
		return nullptr;
	r.lookups.fetch_add(1, std::memory_order_relaxed); // (fase 1f: base de la cobertura)
	// (MALLA-RAPIDA, 11-25) Log 20261125-111241-732: ~1380 usos/frame (2.75 M usos en 1998 frames) pasaban cada uno por
	// shared_mutex + std::map::upper_bound sobre ~15 000 bloques en el hilo de render. Cache de aciertos propia del hilo
	// (8 entradas, por bloque): vale mientras `epoch` no se mueva (toda baja de un VkBuffer y todo alta/baja de bloque la
	// mueve con el mutex exclusivo). Sin candado ni recorrido. La firma de cada SigEvery usos se mantiene: los aciertos se
	// cuentan aqui y el uso numero SigEvery va por el camino de siempre, que la hace.
	// (MALLA-RAPIDA2, 11-25) Log 20261125-112615-361: con 8 entradas solo 1045105 de 7259610 usos (14 %) salian por la
	// cache, y 3076176 huecos pasaban todos por shared_mutex + map. Causas: (1) cada alta de bloque (streaming, ~15 000
	// vivos) movia `epoch` y vaciaba la cache entera; (2) 8 entradas para ~1400 usos/frame repartidos en cientos de
	// bloques. Ahora: tabla directa del hilo de render de 4096 entradas indexada por pagina de 4 KiB del V#; aciertos
	// validos mientras `epoch` no se mueva (solo bajas de VkBuffer/bloque), huecos mientras no se mueva `ins_epoch`
	// tampoco. La firma cada SigEvery usos sigue por el camino de siempre.
	struct Hit {
		uint64_t start = 0, end = 0, page_lo = 0, page_hi = 0, epoch = 0, ins = 0, page = UINT64_MAX;
		void*    gpu = nullptr; // (nullptr con page valida: hueco cacheado, [start, end) sin bloque)
		uint32_t uses = 0;
	};
	constexpr uint32_t    HitSlots = 4096;
	thread_local std::unique_ptr<Hit[]> hits_owner(new Hit[HitSlots]);
	Hit* const            hits  = hits_owner.get();
	thread_local uint32_t carry = 0; // (aciertos que se suman a b.uses en el camino de siempre)
	const uint64_t epoch = r.epoch.load(std::memory_order_acquire);
	const uint64_t ins   = r.ins_epoch.load(std::memory_order_acquire);
	const uint64_t vpage = vaddr >> 12u;
	Hit&           h     = hits[vpage & (HitSlots - 1)];
	if (h.page == vpage && h.epoch == epoch) {
		if (h.gpu != nullptr) {
			if (vaddr >= h.start && vaddr < h.end && vaddr + size <= h.page_hi) {
				if (++h.uses < SigEvery) {
					*page_lo = h.page_lo;
					r.served.fetch_add(1, std::memory_order_relaxed);
					r.fast_hits.fetch_add(1, std::memory_order_relaxed);
					return h.gpu;
				}
				carry  = h.uses - 1; // (este uso, por el camino de siempre: alli llega a SigEvery y se firma)
				h.page = UINT64_MAX;
			}
		} else if (h.ins == ins && vaddr >= h.start && vaddr < h.end) {
			r.miss_gap.fetch_add(1, std::memory_order_relaxed);
			r.fast_gaps.fetch_add(1, std::memory_order_relaxed);
			return nullptr; // (hueco: ningun bloque cubre esta direccion)
		}
	}
	{
		std::shared_lock lock(r.mutex);
		auto             it = r.blocks.upper_bound(vaddr);
		if (it == r.blocks.begin() || vaddr >= std::prev(it)->second.end) { // (fase 1g: hueco, con su origen)
			{ // (MALLA-RAPIDA2) el hueco, recortado a la pagina del V#, cacheado hasta el siguiente alta/baja
				const uint64_t gap_lo = std::max(vpage << 12u, it == r.blocks.begin() ? uint64_t {0} : std::prev(it)->second.end);
				const uint64_t gap_hi = std::min((vpage + 1) << 12u, it == r.blocks.end() ? uint64_t {UINT64_MAX} : it->first);
				h = Hit {gap_lo, gap_hi, 0, 0, epoch, ins, vpage, nullptr, 0};
			}
			lock.unlock();
			r.miss_gap.fetch_add(1, std::memory_order_relaxed);
			if (Diag()) CountGap(vaddr); // (fase 1i: el origen, solo en diagnostico)
			return nullptr;
		}
		--it;
		const auto& b = it->second;
		// (Dentro del bloque; el V# puede pasar del ultimo dato hasta el final de su pagina: el VkBuffer la cubre.)
		if (vaddr + size > b.page_hi) {
			r.miss_over.fetch_add(1, std::memory_order_relaxed);
			return nullptr;
		}
		if (b.failed) {
			r.miss_failed.fetch_add(1, std::memory_order_relaxed);
			return nullptr;
		}
		if (b.gpu != nullptr) {
			// (fase 1i) Cada SigEvery usos: la firma sigue siendo la del VkBuffer? (Lookup: solo el hilo de render)
			if (vaddr + size > b.end) r.served_past_end.fetch_add(1, std::memory_order_relaxed); // (fase 1j, hip. 2)
			const uint32_t add = 1u + carry;
			carry              = 0;
			if (std::atomic_ref<uint32_t>(const_cast<uint32_t&>(b.uses)).fetch_add(add) + add < SigEvery || b.reset_hi != 0) {
				*page_lo = b.page_lo;
				r.served.fetch_add(1, std::memory_order_relaxed);
				if (b.reset_hi == 0) h = Hit {it->first, b.end, b.page_lo, b.page_hi, epoch, ins, vpage, b.gpu, 0}; // (MALLA-RAPIDA2)
				return b.gpu;
			}
			std::atomic_ref<uint32_t>(const_cast<uint32_t&>(b.uses)).store(0, std::memory_order_relaxed);
			const uint64_t start = it->first, gen = b.gen, sig = b.sig;
			uint64_t       now_sig = 0;
			const bool     same = SigOf(b.end - start, &now_sig, [start](uint64_t off, uint8_t* d, uint64_t n) {
                return ReadGuest(start + off, d, n);
            }) && now_sig == sig;
			r.sig_checks.fetch_add(1, std::memory_order_relaxed);
			if (same) {
				*page_lo = b.page_lo;
				r.served.fetch_add(1, std::memory_order_relaxed);
				h = Hit {start, b.end, b.page_lo, b.page_hi, epoch, ins, vpage, b.gpu, 0}; // (MALLA-RAPIDA2)
				return b.gpu;
			}
			lock.unlock();
			std::unique_lock w(r.mutex);
			auto             wi = r.blocks.find(start);
			if (wi != r.blocks.end() && wi->second.gen == gen && wi->second.gpu != nullptr) {
				auto& wb = wi->second;
				NativoCubierto::Descubrir(wi->first, wb.end - wi->first, NativoCubierto::DMallas); // (fase 1l)
				Bury(r, wb); // (BUF-MEM: dedup)
				wb.gpu = nullptr, wb.mapped = nullptr, wb.queued = false;
				r.epoch.fetch_add(1, std::memory_order_release); // (MALLA-RAPIDA)
				r.sig_stale.fetch_add(1, std::memory_order_relaxed);
				if (++wb.stale >= 2) { // (malla que la CPU reescribe: camino normal para siempre)
					wb.failed = true;
					r.sig_off.fetch_add(1, std::memory_order_relaxed);
				}
			}
			return nullptr; // (este uso, por el camino normal: datos actuales)
		}
		if (b.pending == nullptr && b.queued) { // (el Preparer aun no llego: camino normal)
			r.miss_preparing.fetch_add(1, std::memory_order_relaxed);
			return nullptr;
		}
		if (b.pending == nullptr && b.budget_fails != 0 && std::chrono::steady_clock::now() < b.retry_at) {
			r.miss_budget.fetch_add(1, std::memory_order_relaxed); // (fase 1h: esperando su reintento)
			return nullptr;
		}
	}
	std::unique_lock lock(r.mutex); // (OnFree/EraseLocked no pueden soltar `pending` mientras)
	auto             it = r.blocks.upper_bound(vaddr);
	if (it == r.blocks.begin()) return nullptr;
	--it;
	auto& b = it->second;
	if (vaddr >= b.end || vaddr + size > b.page_hi || b.failed) return nullptr;
	if (b.gpu != nullptr) {
		*page_lo = b.page_lo;
		return b.gpu;
	}
	if (b.pending == nullptr) {
		// (primer uso por la GPU: al Preparer; este uso, por el camino normal)
		if (!b.queued) {
			if (b.budget_fails != 0) {
				if (std::chrono::steady_clock::now() < b.retry_at) return nullptr;
				r.retries.fetch_add(1, std::memory_order_relaxed);
			}
			b.queued = true;
			r.requested.fetch_add(1, std::memory_order_relaxed);
			{
				std::lock_guard q(r.queue_mutex);
				r.queue.emplace_back(it->first, b.gen);
			}
			r.queue_cv.notify_one();
		}
		r.miss_first.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	uint8_t* mapped = nullptr;
	bool     vram   = false;
	void*    gpu    = finish(b.pending, b.page_lo, b.page_hi - b.page_lo, &mapped, &vram);
	if (gpu == nullptr) { // (sin prologo ahora: el siguiente uso)
		r.miss_prologue.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	b.pending = nullptr;
	b.gpu     = gpu;
	// (fase 1l) Desde aqui la GPU lee este bloque SOLO por su VkBuffer: sus paginas enteras salen de la vigilancia del
	// emulador. Si la cache del emulador llegara a pedirlas (pasa del bloque, firma distinta), nativo-cubierto.h las
	// descubre solo (SeguridadUsoEmulador) y se cuenta en [vigilancia].
	NativoCubierto::MarcarCubierto(it->first, b.end - it->first, NativoCubierto::DMallas);
	b.mapped  = mapped;
	b.vram    = vram;
	b.idle_s  = 0;
	if (b.dkey != 0 && !r.dedup_by_key.contains(b.dkey)) { // (BUF-MEM: la primera copia de este contenido)
		r.dedup_by_key.emplace(b.dkey, gpu);
		r.dedup_refs[gpu] = Shared {1, b.dkey, b.page_hi - b.page_lo, mapped, vram};
	}
	r.created.fetch_add(1, std::memory_order_relaxed);
	r.created_bytes.fetch_add(b.page_hi - b.page_lo, std::memory_order_relaxed);
	(vram ? r.vram_bytes : r.sys_bytes).fetch_add(b.page_hi - b.page_lo, std::memory_order_relaxed);
	*page_lo = b.page_lo;
	r.served.fetch_add(1, std::memory_order_relaxed);
	return gpu;
}

inline void MarkFailed(uint64_t start, uint64_t gen) noexcept;

// (fase 1h, recolector de texturas) Los bytes que las mallas no pudieron reservar desde la ultima pasada.
inline uint64_t TakeBudgetWant() {
	return On() ? Get().budget_want.exchange(0, std::memory_order_relaxed) : 0;
}

// (hilo Preparer) Lee cada bloque registrado y deja su memoria de GPU preparada (pending). Un fallo marca el bloque
// `failed` para siempre (camino normal, sin reintentos).
inline void PrepareOne(uint64_t start, uint64_t gen) {
	auto&    r   = Get();
	uint64_t plo = 0, phi = 0, end = 0;
	{
		std::shared_lock lock(r.mutex);
		auto             it = r.blocks.find(start);
		if (it == r.blocks.end() || it->second.gen != gen) {
			r.dropped_gone.fetch_add(1, std::memory_order_relaxed); // (malla destruida antes de prepararse)
			return;
		}
		if (it->second.gpu != nullptr || it->second.pending != nullptr || it->second.failed) return;
		plo = it->second.page_lo, phi = it->second.page_hi, end = it->second.end;
	}
	if (phi <= plo || phi - plo > MaxBlock + Page) { // (rango imposible: camino normal, contado)
		r.dropped_big.fetch_add(1, std::memory_order_relaxed);
		MarkFailed(start, gen);
		return;
	}
	const auto           t0 = std::chrono::steady_clock::now();
	std::vector<uint8_t> bytes(phi - plo);
	// (en trozos de 64 KiB: el camino sin candado; entre trozos cede la CPU)
	bool read = true;
	for (uint64_t off = 0; read && off < bytes.size(); off += ReadChunk) {
		const uint64_t n = std::min<uint64_t>(ReadChunk, bytes.size() - off);
		read             = ReadGuest(plo + off, bytes.data() + off, n);
		if (!read) {
			// (fase 1g, el "fallos: lectura 1") El trozo tiene una pagina ilegible. Pagina a pagina: una sin ningun byte
			// del bloque [start, end) (cabeza antes de start o cola redondeada a pagina) va a ceros: la GPU no lee datos de
			// la malla ahi. Una pagina con datos del bloque ilegible: fallo, como antes.
			read = true;
			for (uint64_t p = 0; read && p < n; p += Page) {
				const uint64_t a = plo + off + p;
				if (ReadGuest(a, bytes.data() + off + p, Page)) continue;
				if (a + Page <= start || a >= end) {
					std::memset(bytes.data() + off + p, 0, Page);
					r.read_padded.fetch_add(1, std::memory_order_relaxed);
					continue;
				}
				r.read_fail_addr.store(a, std::memory_order_relaxed);
				read = false;
			}
		}
		if ((off / ReadChunk) % 16 == 15) {
			std::this_thread::yield();
			std::shared_lock lock(r.mutex); // (malla destruida mientras: no se sigue leyendo)
			auto             it = r.blocks.find(start);
			if (it == r.blocks.end() || it->second.gen != gen) return;
		}
	}
	// (BUF-MEM) Dedup: otro bloque vivo con las mismas paginas ya tiene VkBuffer: se comparte, sin reservar nada.
	const uint64_t dkey = read && BufMem::On() ? Fnv(bytes.data(), bytes.size()) ^ (bytes.size() * 0x9e3779b97f4a7c15ull) : 0;
	if (dkey != 0) {
		std::unique_lock lock(r.mutex);
		auto             it = r.blocks.find(start);
		if (it == r.blocks.end() || it->second.gen != gen) return;
		auto& b = it->second;
		b.dkey  = dkey;
		auto k  = r.dedup_by_key.find(dkey);
		if (k != r.dedup_by_key.end() && b.gpu == nullptr && b.pending == nullptr) {
			auto& sh = r.dedup_refs[k->second];
			sh.refs++;
			b.gpu    = k->second;
			b.mapped = sh.mapped;
			b.vram   = sh.vram;
			b.queued = false;
			b.hash   = Fnv(bytes.data() + (start - plo), end - start);
			(void)SigOf(end - start, &b.sig, [&](uint64_t off, uint8_t* d, uint64_t n) {
				std::memcpy(d, bytes.data() + (start - plo) + off, n);
				return true;
			});
			b.uses = b.seen_uses = b.idle_s = 0;
			NativoCubierto::MarcarCubierto(start, end - start, NativoCubierto::DMallas);
			BufMem::g_dedup_bytes.fetch_add(phi - plo, std::memory_order_relaxed);
			BufMem::g_dedup_n.fetch_add(1, std::memory_order_relaxed);
			// (MALLA-TIRON) Sin mover `epoch`: ningun acierto cacheado apunta a este bloque (aun sin VkBuffer).
			return;
		}
	}
	int            reason = read ? FailNone : FailRead;
	void*          prep   = read && r.prepare ? r.prepare(plo, phi - plo, bytes.data(), &reason) : nullptr;
	const uint64_t hash   = prep != nullptr ? Fnv(bytes.data() + (start - plo), end - start) : 0;
	r.prepare_us.fetch_add(static_cast<uint64_t>(
	                           std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
	                       std::memory_order_relaxed);
	std::unique_lock lock(r.mutex);
	auto             it = r.blocks.find(start);
	if (it == r.blocks.end() || it->second.gen != gen) {
		if (prep != nullptr && r.discard) r.discard(prep); // (la malla se destruyo mientras)
		return;
	}
	auto& b = it->second;
	if (prep == nullptr && reason == FailBudget) { // (fase 1h: no fallido; se reintenta y se pide hueco a las texturas)
		b.queued   = false;
		b.retry_at = std::chrono::steady_clock::now() + std::chrono::seconds(RetrySeconds);
		b.budget_fails++;
		r.fail_reason[FailBudget].fetch_add(1, std::memory_order_relaxed);
		const uint64_t want = phi - plo;
		uint64_t       cur  = r.budget_want.load(std::memory_order_relaxed);
		while (cur < (uint64_t {512} << 20u) &&
		       !r.budget_want.compare_exchange_weak(cur, cur + want, std::memory_order_relaxed)) {
		}
		r.budget_want_total.fetch_add(want, std::memory_order_relaxed);
		return;
	}
	if (prep == nullptr) {
		b.failed = true; // (nunca se reintenta: camino normal)
		r.failed.fetch_add(1, std::memory_order_relaxed);
		r.fail_reason[reason > FailNone && reason < FailCount ? reason : FailAlloc].fetch_add(1, std::memory_order_relaxed);
		return;
	}
	b.pending = prep;
	b.hash    = hash;
	(void)SigOf(end - start, &b.sig, [&](uint64_t off, uint8_t* d, uint64_t n) { // (fase 1j: tres ventanas)
		std::memcpy(d, bytes.data() + (start - plo) + off, n);
		return true;
	});
	b.uses    = 0;
	r.prepared.fetch_add(1, std::memory_order_relaxed);
}

// Una excepcion en un hilo propio: aviso (las primeras 16) y el hilo sigue.
inline void Warn(const char* where, const char* what) noexcept {
	static std::atomic<uint32_t> n {0};
	if (n.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("[malla-nativa] AVISO: excepcion en %s: %s (capturada; el bloque va por el camino normal)\n", where,
		            what != nullptr ? what : "?");
		std::fflush(stdout);
	}
}

inline void MarkFailed(uint64_t start, uint64_t gen) noexcept {
	try {
		auto&            r = Get();
		std::unique_lock lock(r.mutex);
		auto             it = r.blocks.find(start);
		if (it == r.blocks.end() || it->second.gen != gen || it->second.gpu != nullptr || it->second.pending != nullptr) return;
		it->second.failed = true;
		r.failed.fetch_add(1, std::memory_order_relaxed);
	} catch (...) {
	}
}

inline void StartPreparer() {
	std::thread([] {
		auto& r = Get();
#if defined(_WIN32)
		SetThreadPriority(GetCurrentThread(), -2); // (THREAD_PRIORITY_LOWEST: no compite con el render ni el juego)
#endif
		auto     window = std::chrono::steady_clock::now();
		uint64_t window_bytes = 0;
		for (;;) try {
			std::pair<uint64_t, uint64_t> job;
			{
				std::unique_lock q(r.queue_mutex);
				r.queue_cv.wait(q, [&] { return !r.queue.empty(); });
				job = r.queue.front();
				r.queue.pop_front();
			}
			while (!r.backend.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
			uint64_t bytes = 0;
			{
				std::shared_lock lock(r.mutex);
				auto             it = r.blocks.find(job.first);
				if (it != r.blocks.end()) bytes = it->second.page_hi - it->second.page_lo;
			}
			// (limite: 32 MiB por segundo)
			auto now = std::chrono::steady_clock::now();
			if (now - window >= std::chrono::seconds(1)) window = now, window_bytes = 0;
			if (window_bytes > 0 && window_bytes + bytes > RateBytes) {
				std::this_thread::sleep_until(window + std::chrono::seconds(1));
				window = std::chrono::steady_clock::now(), window_bytes = 0;
			}
			window_bytes += bytes;
			try {
				PrepareOne(job.first, job.second);
			} catch (const std::exception& e) {
				Warn("Preparer", e.what());
				MarkFailed(job.first, job.second);
			} catch (...) {
				Warn("Preparer", "?");
				MarkFailed(job.first, job.second);
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
		} catch (const std::exception& e) {
			Warn("Preparer (cola)", e.what());
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		} catch (...) {
			Warn("Preparer (cola)", "?");
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}).detach();
}

// (BufferCache, una vez) Las funciones del Preparer.
inline void SetBackend(std::function<void*(uint64_t, uint64_t, const uint8_t*, int*)> prepare, std::function<void(void*)> discard) {
	auto& r = Get();
	if (r.backend.load(std::memory_order_acquire)) return;
	r.prepare = std::move(prepare);
	r.discard = std::move(discard);
	r.backend.store(true, std::memory_order_release);
}

// (hilo de render) Los VkBuffer de mallas destruidas.
inline std::vector<void*> TakeGraveyard() {
	auto&              r = Get();
	std::vector<void*> out;
	if (!r.dead.exchange(false, std::memory_order_acq_rel)) return out;
	std::unique_lock lock(r.mutex);
	out.swap(r.graveyard);
	return out;
}
// (fase 1j) Devuelve a la fosa lo que aun no se puede destruir de forma diferida (sin planificador activo).
inline void KeepGraveyard(void* gpu) {
	auto&            r = Get();
	std::unique_lock lock(r.mutex);
	r.graveyard.push_back(gpu);
	r.dead.store(true, std::memory_order_release);
	r.grave_kept.fetch_add(1, std::memory_order_relaxed);
}

// (hilo propio, cada segundo) Fase 2 y la linea de cada 30 s.
inline void Tick() {
	auto&      r   = Get();
	const auto now = std::chrono::steady_clock::now();
	if (RamOn() && r.read_hooked.load(std::memory_order_acquire)) { // (sin los accesores enganchados, nada se suelta)
		// Candidatos: con VkBuffer en VRAM, sin soltar, sin lecturas de CPU en 30 s, no modificados.
		std::vector<std::pair<uint64_t, uint64_t>> cands; // (inicio, gen)
		{
			std::shared_lock lock(r.mutex);
			for (const auto& [s, b]: r.blocks) {
				if (b.gpu == nullptr || !b.vram || b.mapped == nullptr || b.reset_hi != 0 || b.modified) continue;
				// (fase 2b) Solo mallas que la CPU no toca: nunca leidas por 0x909940/0x909970 (8+18 llamadores; los de
				// 0x909940 escriben en anillos de otros objetos) y sin ninguna firma distinta (stale).
				if (b.read_by_cpu || b.stale != 0) continue;
				if (now - b.last_read < std::chrono::seconds(IdleSeconds)) continue;
				const uint64_t lo = (s + Page - 1) & ~(Page - 1); // (fase 1k: paginas enteras DENTRO del bloque)
				const uint64_t hi = b.end & ~(Page - 1);
				if (hi <= lo) continue;
				cands.emplace_back(s, b.gen);
				if (cands.size() >= 64) break; // (el resto en el segundo siguiente)
			}
		}
		std::vector<uint8_t> tmp;
		for (const auto& [s, gen]: cands) {
			std::lock_guard ram(r.ram_mutex); // (ninguna lectura ni destruccion mientras)
			uint64_t        end = 0, hash = 0;
			{
				std::shared_lock lock(r.mutex);
				auto             it = r.blocks.find(s);
				if (it == r.blocks.end() || it->second.gen != gen || it->second.reset_hi != 0 || it->second.read_by_cpu ||
				    it->second.stale != 0 || it->second.gpu == nullptr || now - it->second.last_read < std::chrono::seconds(IdleSeconds))
					continue;
				end = it->second.end, hash = it->second.hash;
			}
			// Los bytes del juego siguen siendo los subidos? (si la CPU los reescribio, no se suelta nunca)
			tmp.resize(end - s);
			if (!ReadGuest(s, tmp.data(), tmp.size())) continue;
			auto it = r.blocks.find(s); // (ram_mutex: el bloque sigue)
			if (it == r.blocks.end() || it->second.gen != gen) continue;
			auto& b = it->second;
			if (b.read_by_cpu || b.gpu == nullptr) continue; // (leida mientras se comparaba)
			if (Fnv(tmp.data(), tmp.size()) != hash) {
				std::unique_lock lock(r.mutex);
				b.modified = true;
				r.modified.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			const uint64_t lo = (s + Page - 1) & ~(Page - 1);
			const uint64_t hi = end & ~(Page - 1);
			uint64_t       refused = 0;
			const uint64_t done    = Libs::LibKernel::Memory::NativeOffloadDrop(lo, hi - lo, &refused); // (sin mutex)
			if (done == 0 || refused != 0) {
				// (algo no se solto: se cuenta; si se solto parte, se restaura entero al leer)
				r.reset_refused.fetch_add(1, std::memory_order_relaxed);
				if (done == 0) continue;
			}
			std::unique_lock lock(r.mutex);
			b.reset_lo = lo, b.reset_hi = hi;
			r.reset_bytes.fetch_add(hi - lo, std::memory_order_relaxed);
			r.resets.fetch_add(1, std::memory_order_relaxed);
		}
	}
	// (BUF-MEM) Streaming: un bloque con VkBuffer sin usos de la GPU en StreamOutSeconds sale de VRAM. Si sus paginas de
	// RAM estaban soltadas (fase 2), primero vuelven (Restore: la unica copia pasa a la RAM del juego); despues el VkBuffer
	// a la fosa. El siguiente uso lo vuelve a preparar el hilo Preparer (asincrono; ese uso, por el camino normal).
	if (BufMem::On()) {
		std::vector<std::pair<uint64_t, uint64_t>> idle; // (inicio, gen)
		{
			std::unique_lock lock(r.mutex);
			for (auto& [s, b]: r.blocks) {
				if (b.gpu == nullptr || !b.vram) continue;
				if (b.uses != b.seen_uses) {
					b.seen_uses = b.uses, b.idle_s = 0;
					continue;
				}
				if (++b.idle_s >= static_cast<uint32_t>(BufMem::StreamOutSeconds)) idle.emplace_back(s, b.gen);
				if (idle.size() >= 32) break;
			}
		}
		for (const auto& [s, gen]: idle) {
			std::lock_guard ram(r.ram_mutex);
			auto            it = r.blocks.find(s); // (ram_mutex: nadie inserta ni borra)
			if (it == r.blocks.end() || it->second.gen != gen || it->second.gpu == nullptr) continue;
			auto& b = it->second;
			if (b.reset_hi != 0) Restore(r, s, b, true); // (toma mutex dentro)
			std::unique_lock lock(r.mutex);
			if (b.gpu == nullptr || b.reset_hi != 0) continue; // (reescrita por la CPU: Restore ya la quito)
			const uint64_t bytes = b.page_hi - b.page_lo;
			NativoCubierto::Descubrir(s, b.end - s, NativoCubierto::DMallas);
			Bury(r, b);
			b.gpu = nullptr, b.mapped = nullptr, b.queued = false, b.idle_s = 0, b.seen_uses = b.uses = 0;
			r.epoch.fetch_add(1, std::memory_order_release);
			BufMem::g_mesh_out.fetch_add(bytes, std::memory_order_relaxed);
			BufMem::g_mesh_out_n.fetch_add(1, std::memory_order_relaxed);
		}
	}
	static auto last = now;
	if (now - last >= std::chrono::seconds(30)) {
		last = now;
		const uint64_t lk = r.lookups.load(), sv = r.served.load();
		// (fase 1k) Resumen: % de usos de la GPU en el rango de mallas servidos en nativo y MiB de RAM devueltos.
		std::printf("[malla-nativa] RESUMEN: usos nativos %.1f %% (%llu de %llu; el resto por el emulador: huecos %llu = bufers "
		            "por dibujo/fotograma o escritos por la GPU que NO son mallas, ver cabecera fase 1k); RAM devuelta %.1f MiB "
		            "(%s, accesores 0x909940/0x909970 %s)\n",
		            lk != 0 ? 100.0 * static_cast<double>(sv) / static_cast<double>(lk) : 0.0, static_cast<unsigned long long>(sv),
		            static_cast<unsigned long long>(lk), static_cast<unsigned long long>(r.miss_gap.load()),
		            static_cast<double>(r.reset_bytes.load()) / (1024.0 * 1024.0),
		            RamOn() ? "encendido por defecto; KYTY_MALLA_NATIVA_RAM=0 lo apaga" : "apagado con KYTY_MALLA_NATIVA_RAM=0",
		            r.read_hooked.load(std::memory_order_acquire) ? "enganchados" : "NO enganchados: no se suelta nada");
		std::printf("[malla-nativa] cobertura %.1f %% (usos servidos por VkBuffer de malla %llu de %llu en el rango de mallas); "
		            "bloques borrados por solape %llu, descartados: malla destruida antes %llu, demasiado grandes %llu\n",
		            lk != 0 ? 100.0 * static_cast<double>(sv) / static_cast<double>(lk) : 0.0, static_cast<unsigned long long>(sv),
		            static_cast<unsigned long long>(lk), static_cast<unsigned long long>(r.overlapped.load()),
		            static_cast<unsigned long long>(r.dropped_gone.load()), static_cast<unsigned long long>(r.dropped_big.load()));
		std::printf("[malla-nativa] estado: encendido (KYTY_MALLA_NATIVA=0 lo apaga), diagnostico %s (gancho CBuffer::Create 0x909460 %s); firma cada %u usos: "
		            "comprobaciones %llu, datos viejos descartados %llu, bloques pasados al camino normal %llu\n",
		            Diag() ? "ENCENDIDO (KYTY_MALLA_NATIVA_DIAG=1)" : "apagado", Diag() ? "puesto" : "NO puesto", SigEvery,
		            static_cast<unsigned long long>(r.sig_checks.load()), static_cast<unsigned long long>(r.sig_stale.load()),
		            static_cast<unsigned long long>(r.sig_off.load()));
		std::printf("[malla-nativa] fase 1j: firma de 3 ventanas (inicio/medio/final); servidos que pasan del final de la malla %llu, "
		            "usos en traduccion especulativa/grabador al camino normal %llu, VkBuffer de la fosa retenidos sin planificador %llu\n",
		            static_cast<unsigned long long>(r.served_past_end.load()), static_cast<unsigned long long>(r.spec_skipped.load()),
		            static_cast<unsigned long long>(r.grave_kept.load()));
		{
			// (fase 1g) Los no servidos por motivo y los llamadores de CBuffer::Create (0x909460) de los huecos.
			std::printf("[malla-nativa] no servidos: hueco %llu (origen conocido %llu), pasa del bloque %llu, bloque fallido %llu, "
			            "preparandose %llu, primer uso %llu, sin prologo %llu; paginas sin datos ilegibles a ceros %llu, "
			            "ultima direccion ilegible con datos 0x%llx\n",
			            static_cast<unsigned long long>(r.miss_gap.load()), static_cast<unsigned long long>(r.miss_gap_known.load()),
			            static_cast<unsigned long long>(r.miss_over.load()), static_cast<unsigned long long>(r.miss_failed.load()),
			            static_cast<unsigned long long>(r.miss_preparing.load()), static_cast<unsigned long long>(r.miss_first.load()),
			            static_cast<unsigned long long>(r.miss_prologue.load()), static_cast<unsigned long long>(r.read_padded.load()),
			            static_cast<unsigned long long>(r.read_fail_addr.load()));
			std::vector<std::pair<uint64_t, uint64_t>> top; // (usos, indice)
			auto*                                      t = Callers();
			for (uint64_t i = 0; Diag() && i < CallerSlots; i++) // (fase 1i: llamadores solo en diagnostico)
				if (t[i].caller.load() != 0) top.emplace_back(t[i].uses.load(), i);
			std::sort(top.begin(), top.end(), std::greater<>());
			for (size_t k = 0; k < top.size() && k < 10; k++) {
				const auto&    c   = t[top[k].second];
				const uint64_t ret = c.caller.load() - 1, uses = c.uses.load();
				std::printf("[malla-nativa]   hueco de eboot+0x%llx (retorno de CBuffer::Create; Ghidra archivo 0x%llx): usos %llu, "
				            "flags 0x%llx, bytes medios %llu\n",
				            static_cast<unsigned long long>(ret), static_cast<unsigned long long>(ret + 0xbdb0),
				            static_cast<unsigned long long>(uses), static_cast<unsigned long long>(c.flags.load()),
				            static_cast<unsigned long long>(uses != 0 ? c.bytes.load() / uses : 0));
			}
		}
		uint64_t blocks = 0, read = 0, kept = 0;
		{
			std::shared_lock lock(r.mutex);
			blocks = r.blocks.size();
			for (const auto& [s, b]: r.blocks) {
				read += b.read_by_cpu ? 1 : 0;
				kept += b.gpu != nullptr && (b.read_by_cpu || b.stale != 0) ? 1 : 0; // (fase 2b)
			}
		}
		constexpr double MiB = 1024.0 * 1024.0;
		std::printf("[malla-nativa] mallas %llu (bloques vivos %llu, destruidas %llu); en VkBuffer %llu (%.1f MB VRAM, %.1f MB en RAM "
		            "del sistema: sin VRAM visible), fallidos %llu, usos por la GPU %llu; leidas por CPU %llu bloques (vivos %llu); "
		            "pedidos por la GPU %llu, preparados por el hilo propio %llu (%.1f ms, %llu con staging); "
		            "fallos: lectura %llu, vkCreateBuffer %llu, sin presupuesto %llu, reserva VMA %llu, staging %llu; "
		            "RAM devuelta %.1f MB%s (soltados %llu, restaurados %llu en %.1f ms, reescritos por la CPU %llu, rechazos %llu); "
		            "reserva %llu MB (creados dentro %llu), reintentos %llu (usos esperando %llu), hueco pedido a texturas %.1f MB\n",
		            static_cast<unsigned long long>(r.meshes.load()), static_cast<unsigned long long>(blocks),
		            static_cast<unsigned long long>(r.freed.load()), static_cast<unsigned long long>(r.created.load()),
		            static_cast<double>(r.vram_bytes.load()) / MiB, static_cast<double>(r.sys_bytes.load()) / MiB,
		            static_cast<unsigned long long>(r.failed.load()), static_cast<unsigned long long>(r.served.load()),
		            static_cast<unsigned long long>(r.cpu_reads.load()), static_cast<unsigned long long>(read),
		            static_cast<unsigned long long>(r.requested.load()),
		            static_cast<unsigned long long>(r.prepared.load()), static_cast<double>(r.prepare_us.load()) / 1000.0,
		            static_cast<unsigned long long>(r.staged.load()),
		            static_cast<unsigned long long>(r.fail_reason[FailRead].load()),
		            static_cast<unsigned long long>(r.fail_reason[FailProbe].load()),
		            static_cast<unsigned long long>(r.fail_reason[FailBudget].load()),
		            static_cast<unsigned long long>(r.fail_reason[FailAlloc].load()),
		            static_cast<unsigned long long>(r.fail_reason[FailStaging].load()),
		            static_cast<double>(r.reset_bytes.load()) / MiB, RamOn() ? "" : " (apagado: KYTY_MALLA_NATIVA_RAM=0)",
		            static_cast<unsigned long long>(r.resets.load()), static_cast<unsigned long long>(r.restores.load()),
		            static_cast<double>(r.restore_us.load()) / 1000.0, static_cast<unsigned long long>(r.modified.load()),
		            static_cast<unsigned long long>(r.reset_refused.load()),
		            static_cast<unsigned long long>(ReserveBytes() >> 20u), static_cast<unsigned long long>(r.in_reserve.load()),
		            static_cast<unsigned long long>(r.retries.load()), static_cast<unsigned long long>(r.miss_budget.load()),
		            static_cast<double>(r.budget_want_total.load()) / MiB);
		// (fase 2b) La prueba de que lo soltado no tenia escritores: reescritos TRAS soltar (comprobado pagina a pagina en
		// cada restauracion y al destruir la malla) debe ser 0.
		std::printf("[malla-nativa] RAM fase 2b: %.1f MB devueltos; reescritos por la CPU TRAS soltar %llu bloques (%llu paginas; "
		            "debe ser 0), antes de soltar %llu; mallas en VkBuffer que se quedan en RAM (leidas por la CPU o firma distinta) %llu\n",
		            static_cast<double>(r.reset_bytes.load()) / MiB, static_cast<unsigned long long>(r.rewritten_after.load()),
		            static_cast<unsigned long long>(r.rewritten_pages.load()), static_cast<unsigned long long>(r.modified.load()),
		            static_cast<unsigned long long>(kept));
		std::printf("[malla-rapida] usos servidos sin candado ni mapa (cache del hilo de render) %llu de %llu; huecos resueltos sin candado %llu de %llu\n", static_cast<unsigned long long>(r.fast_hits.load()), static_cast<unsigned long long>(r.served.load()), static_cast<unsigned long long>(r.fast_gaps.load()), static_cast<unsigned long long>(r.miss_gap.load()));
		std::fflush(stdout);
	}
}

inline void StartWorker() {
	auto& r = Get();
	if (r.worker.exchange(true)) return;
	try {
		StartPreparer();
		std::thread([] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			try {
				Tick();
			} catch (const std::exception& e) {
				Warn("Tick", e.what());
			} catch (...) {
				Warn("Tick", "?");
			}
		}
		}).detach();
	} catch (const std::exception& e) {
		Warn("StartWorker", e.what()); // (sin hilos: Lookup encola y nadie prepara; todo por el camino normal)
	} catch (...) {
		Warn("StartWorker", "?");
	}
}

} // namespace MallaNativa
