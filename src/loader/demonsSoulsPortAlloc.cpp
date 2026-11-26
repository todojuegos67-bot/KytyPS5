// (p1, 11-25) KYTY_PORT_ALLOC: asignador de paginas nativo de Windows en lugar del PlatformPageAllocator de PS5 del juego.
//
// Ghidra, eboot 01.007.000 (base 0). Todo el asignador vive en 0x80c200..0x810e90 y solo se entra por:
//   0x80d640 Alloc(alloc, size, align, memtype, flags*)            -> cerrojo [0x2ee4b60] -> 0x80d9a0(alloc,0,0,size,0,...)
//   0x80e620 RequestVirtualPages(alloc, info, addr, size, type,...) -> cerrojo            -> 0x80d9a0(todo)
//   0x80d7b0 (sin cerrojo, misma llamada que 0x80d640)
//   0x80f550 Free(info)                                            (toma el cerrojo; llamado desde 0x8019a1, 0x801a4b,
//                                                                   0x8c7e44)
// 0x80d9a0 = RequestVirtualPages_NoThreadSafety(IAllocator* a, PageAllocationInfo* info, void* addr, size_t size,
//            EVirtualAllocType type, EAlignment align, EMemoryType memtype, const TPageAllocationFlags* flags):
//   info == 0, type 0: bloque nuevo: reserva (ReserveVirtualRange, alineacion max(align, 1 MiB), primero en cualquier
//                      sitio y si no en 0x1000000000) + paginas fisicas del mapa de bits + cola de BatchMap (prot 0xf3).
//   info == 0, type 1: solo reserva (descriptor con +0x50 = 0).
//   info != 0, type 0: compromete [addr, addr + size) dentro de info (info+0x50 += size, pool+0x90 += size).
//   info != 0, type 2: descompromete [addr, addr + size) (pool+0x90 -= size; UnmapPhysicalPages 0x80eb90 deja la
//                      reserva: sceKernelVirtualQuery + ReserveVirtualRange FIXED).
//   Devuelve el descriptor (info si ya existia) o 0 = sin memoria (el que llama hace HandleOutOfMemory 0x7f9890).
// Descriptor (0x78 bytes, los reparte 0x80d0a0): +0x00..+0x18 nodo de la lista del que llama (lo rellena el), +0x20 nodo
//   del pool {dueno, +0x28 anterior, +0x30 siguiente}, +0x38 pool, +0x40 IAllocator, +0x48 -1, +0x50 bytes comprometidos,
//   +0x58 tamano, +0x60 direccion, +0x70 indice de pool (byte, = pool+0x78), +0x71 1 = en uso.
// Estado *(0x2ee4d60): pools de 0xa0 (0 y 1; los bloques nuevos siempre en el 1, +0xa0): +0x00 centinela de en uso
//   (+0x08 cola), +0x20 bloques en uso, +0x28 centinela de libres, +0x48 libres, +0x70 / +0x74 paginas libres (mapeadas /
//   sin mapear), +0x78 indice, +0x80 / +0x88 tamano en uso / pico, +0x90 / +0x98 comprometido / pico.
//   Tablas de dueno (las lee 0x80d5a0 para saber de que IAllocator es una direccion): byte IAllocator+0x30 en
//   estado+0x15fa68 + (dir >> 20) si el indice del pool != 0; si no, estado+0x25ba78 + ((dir - [estado+0x25ba68]) >> 16).
//   0x80f550 no las borra. Contadores globales: [0x2ee4d50] +1 por operacion (Free lo suma el mismo; Alloc el envoltorio).
//
// Sustitucion: 0x80d9a0 entero (entrada -> AllocHook) y 0x80f550 para los descriptores de aqui (entrada -> FreeHook; uno
// ajeno sigue por el original). Memoria: cada reserva es un rango de direcciones del juego (sceKernelReserveVirtualRange
// del emulador: marcador de posicion de Windows), cada compromiso memoria privada de Windows en esa direccion
// (VirtualAlloc2 sobre el marcador, comprometida al tocar con commit-real) registrada en la tabla de memoria del emulador
// para que la GPU emulada la vea (TryGetBackingPointer, vigilancia de escrituras, VirtualQuery); al liberar,
// MEM_DECOMMIT y fuera. Sin mapa de bits, sin cola de BatchMap, sin listas de bloques mapeados libres, sin pool de 12 GiB.
// La clave fisica que pide la tabla del emulador sale de sceKernelAllocateDirectMemory por bloque (no es memoria: en
// memoria nativa del emulador una clave sin mapear no ocupa nada), troceada si hace falta: sin limite de contiguidad.
#include "loader/demonsSoulsPortAlloc.h"

#include "common/abi.h"
#include "common/guardedCopy.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsPageFree.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "local/local-platform.h"
#include "ram-guard.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "guest-lazy-commit.h" // (NativeLazyCommit; incluye windows.h y psapi.h en Windows)

namespace Loader::DemonsSoulsPortAlloc {
namespace {

namespace PF  = Loader::DemonsSoulsPageFree;
namespace Mem = Libs::LibKernel::Memory;

// Entrada de 0x80d9a0 (01.007.000; unica en el eboot): push rbp ... sub rsp,0x140; mov rax,[rip+..]; mov r15d,r9d; ...
constexpr std::string_view RequestPages = "55 48 89 e5 41 57 41 56 41 55 41 54 53 48 83 e4 e0 48 81 ec 40 01 00 00 48 8b 05 ?? ?? "
                                          "?? ?? 45 89 cf 49 89 d2 48 8b 55 18 49 89 c9 49 89 fe 41 81 c7 ff ff 00 00 41 81 e7 00 "
                                          "00 ff ff 45 85 c0";
constexpr size_t JumpSize = 12; // mov rax, imm64; jmp rax
// Prologo de 0x80f550 que va al trampolin: push rbp; mov rbp,rsp; push r15; push r14; push rbx; sub rsp,0x18 (13 bytes).
constexpr size_t FreePrologue = 13;

// Descriptor (PageAllocationInfo).
constexpr uint64_t DescSize = 0x78, DNode = 0x20, DPrev = 0x28, DNext = 0x30, DPool = 0x38, DAlloc = 0x40, DIndex48 = 0x48,
                   DMapped = 0x50, DSize = 0x58, DAddr = 0x60, DPoolIdx = 0x70, DInUse = 0x71;
// Pool (0xa0 desde el estado).
constexpr uint64_t PoolStride = 0xa0, PTail = 0x08, PBlocks = 0x20, PFreeMapped = 0x70, PFreeUnmapped = 0x74, PIdx = 0x78,
                   PSpan = 0x80, PSpanPeak = 0x88, PMapped = 0x90, PMappedPeak = 0x98;
// Tablas de dueno y ventana de 64 KiB.
constexpr uint64_t OwnerMiB = 0x15fa68, OwnerMiBCount = 0xfc000, OwnerWin = 0x25ba78, WinBase = 0x25ba68, WinSize = 0x25ba70;
constexpr uint64_t AllocatorIdAt = 0x30;
// [0x2ee4d50] (operaciones) = [0x2ee4d60] - 0x10.
constexpr uint64_t OpsFromState = 0x10;

constexpr uint64_t Page = 0x10000, MiB = 0x100000, Hint = 0x1000000000ull;
constexpr int      ProtGame = 0xf3, MapFixed = 0x10;

using FreeFn = void(KYTY_SYSV_ABI*)(uint64_t);

struct Chunk {
	uint64_t len  = 0;
	int64_t  phys = 0;
};
struct Block {
	uint64_t                  addr = 0, size = 0;
	std::map<uint64_t, Chunk> chunks; // direccion -> trozo comprometido
};

std::mutex                          g_mutex; // (bloques de aqui; el cerrojo del juego ya ordena las llamadas)
std::unordered_map<uint64_t, Block> g_blocks; // descriptor -> bloque
std::vector<uint8_t*>               g_free_descs;

uint64_t g_alloc_site = 0, g_free_site = 0, g_lock = 0, g_state_global = 0, g_budget = 0;
FreeFn   g_original_free = nullptr;

std::atomic<bool>     g_stop {false}, g_running {false}, g_active {false};
std::atomic<uint64_t> g_reserved {0}, g_mapped {0}, g_peak {0}, g_allocs {0}, g_frees {0}, g_fails {0}, g_decommitted {0},
    g_key_splits {0};

template <typename T>
T& At(uint64_t address) {
	return *reinterpret_cast<T*>(address);
}

uint64_t State() {
	return At<uint64_t>(g_state_global);
}

void NotePeak() {
	const uint64_t now  = g_mapped.load(std::memory_order_relaxed);
	uint64_t       peak = g_peak.load(std::memory_order_relaxed);
	while (now > peak && !g_peak.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
	}
}

// Presupuesto que ve el juego (0x80d4d0 / 0x80d540: (+0x70 + +0x74 de los pools) * 64 KiB): solo un numero, nadie lo hace
// cumplir aqui. KYTY_PORT_ALLOC_PRESUPUESTO_MB (por defecto 12288, lo que el juego espera de una PS5).
void PublishBudget(uint64_t state) {
	const uint64_t pool = state + PoolStride;
	const uint64_t used = At<uint64_t>(pool + PMapped);
	const uint64_t free = g_budget > used ? g_budget - used : 0;
	At<int32_t>(pool + PFreeMapped)   = 0;
	At<int32_t>(pool + PFreeUnmapped) = static_cast<int32_t>(std::min<uint64_t>(free >> 16u, 0x7fffffff));
}

uint8_t* NewDescriptor() {
	uint8_t* d = nullptr;
	if (!g_free_descs.empty()) {
		d = g_free_descs.back();
		g_free_descs.pop_back();
	} else {
		d = static_cast<uint8_t*>(std::calloc(1, 0x80));
	}
	if (d != nullptr) std::memset(d, 0, 0x80);
	return d;
}

// Compromete [va, va + len) (dentro de una reserva propia): claves fisicas del emulador + mapeo FIXED.
bool CommitRange(Block* b, uint64_t va, uint64_t len) {
	const auto dms = static_cast<int64_t>(Mem::KernelGetDirectMemorySize());
	uint64_t   at = va, left = len, piece = len;
	while (left != 0) {
		piece       = std::min(piece, left);
		int64_t key = 0;
		if (Mem::KernelAllocateDirectMemory(0, dms, piece, Page, 0, &key) != 0) {
			if (piece <= Page) return false;
			piece = std::max<uint64_t>(Page, (piece / 2u) & ~(Page - 1u)); // (trozos: sin limite de contiguidad)
			g_key_splits.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		void* where = reinterpret_cast<void*>(at);
		if (Mem::KernelMapDirectMemory(&where, piece, ProtGame, MapFixed, key, 0) != 0 || reinterpret_cast<uint64_t>(where) != at) {
			(void)Mem::KernelReleaseDirectMemory(key, piece);
			return false;
		}
		b->chunks[at] = {piece, key};
		at += piece;
		left -= piece;
		g_mapped.fetch_add(piece, std::memory_order_relaxed);
	}
	NotePeak();
	return true;
}

// Devuelve a Windows [va, va + len) de un trozo y lo deja reservado (keep_reserved) o sin nada.
void DropPart(uint64_t va, uint64_t len, int64_t key, bool keep_reserved) {
	(void)Mem::TryPrepareHostOverwrite(va, len); // (copias de la GPU fuera: si se vuelve a usar, se suben de nuevo)
#if defined(_WIN32)
	uint64_t rest = 0;
	g_decommitted.fetch_add(NativeLazyCommit::DecommitFree(va, len, &rest), std::memory_order_relaxed);
#endif
	if (keep_reserved) {
		void* where = reinterpret_cast<void*>(va);
		(void)Mem::KernelReserveVirtualRange(&where, len, MapFixed, 0);
	} else {
		(void)Mem::KernelMunmap(va, len);
	}
	(void)Mem::KernelReleaseDirectMemory(key, len);
	g_mapped.fetch_sub(std::min(len, g_mapped.load(std::memory_order_relaxed)), std::memory_order_relaxed);
}

// Descompromete [va, va + len); devuelve los bytes que estaban comprometidos.
uint64_t DecommitRange(Block* b, uint64_t va, uint64_t len) {
	const uint64_t end  = va + len;
	uint64_t       done = 0;
	auto           it   = b->chunks.upper_bound(va);
	if (it != b->chunks.begin()) --it;
	while (it != b->chunks.end() && it->first < end) {
		const uint64_t c_begin = it->first, c_end = it->first + it->second.len;
		const int64_t  key     = it->second.phys;
		if (c_end <= va) {
			++it;
			continue;
		}
		const uint64_t cut_begin = std::max(va, c_begin), cut_end = std::min(end, c_end);
		it = b->chunks.erase(it);
		if (c_begin < cut_begin) b->chunks[c_begin] = {cut_begin - c_begin, key};
		if (cut_end < c_end) b->chunks[cut_end] = {c_end - cut_end, key + static_cast<int64_t>(cut_end - c_begin)};
		DropPart(cut_begin, cut_end - cut_begin, key + static_cast<int64_t>(cut_begin - c_begin), true);
		done += cut_end - cut_begin;
		it = b->chunks.lower_bound(cut_end);
	}
	return done;
}

// (c67, 11-26) KYTY_PORT_ALLOC_COLCHON=1: colchon de bloques ya comprometidos.
// Ghidra 0x80f550 (Free original): si el bloque esta entero mapeado (+0x58 == +0x50) NO llama a 0x80eb90 (unmap): lo
// pone en la lista de libres del pool (+0x28, +0x48 += 1), fusionando con vecinos, con sus paginas fisicas puestas; solo
// si esta a medias lo desmapea (0x80eb90(...,1)). 0x80d9a0 (info == 0) saca los bloques nuevos de esa lista
// (0x80e7a0) sin poner a ceros (ni 0x80d9a0 ni 0x80f550 escriben el contenido): el juego ya reutiliza memoria vieja sin
// ceros en PS5, asi que aqui tampoco se pone a ceros. Al liberar, un bloque entero comprometido va al colchon (por
// tamano y zona) en vez de a Windows; la siguiente peticion type 0 del mismo tamano y alineacion lo reutiliza (mismas
// tablas de dueno y lista de en uso que un bloque nuevo; sin cola de BatchMap, como la ruta nativa).
// Tope KYTY_PORT_ALLOC_COLCHON_MB (256); todo a Windows con presion de RAM (RamGuard: LowMemory o < 256 MiB
// disponibles); cada bloque a Windows tras KYTY_PORT_ALLOC_COLCHON_S (10) s sin uso.
struct Cushion {
	Block    b;
	bool     vm   = false;
	uint64_t t_ms = 0;
};
std::list<Cushion>    g_cushion; // (bajo g_mutex; el mas viejo delante)
uint64_t              g_cushion_cap = 0, g_cushion_age_ms = 10000;
bool                  g_cushion_on  = false;
std::atomic<uint64_t> g_cushion_bytes {0}, g_cushion_n {0}, g_cushion_reused {0}, g_cushion_drop_pressure {0},
    g_cushion_drop_age {0};

uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool RamPressure() {
	const uint64_t avail = RamGuard::AvailablePhysical();
	return RamGuard::PressureLevel() != RamGuard::None || (avail != 0 && avail < (uint64_t {256} << 20u)); // (11-26) 512 -> 256 MiB: prueba 68, con 512 se vaciaba siempre
}

bool FullyCommitted(const Block& b) {
	uint64_t at = b.addr;
	for (const auto& [va, c]: b.chunks) {
		if (va != at) return false;
		at += c.len;
	}
	return at == b.addr + b.size;
}

// Bloque entero a Windows (bajo g_mutex): cada trozo a Windows y reservado; luego la reserva entera fuera (sin huecos: el
// munmap del emulador recorre rangos seguidos y se para en el primer hueco).
void ReleaseBlock(Block* b) {
	for (const auto& [va, c]: b->chunks) DropPart(va, c.len, c.phys, true);
	b->chunks.clear();
	(void)Mem::KernelMunmap(b->addr, b->size);
	g_reserved.fetch_sub(std::min(b->size, g_reserved.load(std::memory_order_relaxed)), std::memory_order_relaxed);
}

void CushionDrop(std::list<Cushion>::iterator it, std::atomic<uint64_t>* counter) {
	const uint64_t size = it->b.size;
	ReleaseBlock(&it->b);
	g_cushion_bytes.fetch_sub(std::min(size, g_cushion_bytes.load(std::memory_order_relaxed)), std::memory_order_relaxed);
	g_cushion_n.fetch_sub(1, std::memory_order_relaxed);
	if (counter != nullptr) counter->fetch_add(size, std::memory_order_relaxed);
	g_cushion.erase(it);
}

// Con presion: todo fuera; si no, los viejos (bajo g_mutex).
void CushionTrim(uint64_t now) {
	if (g_cushion.empty()) return;
	if (RamPressure()) {
		while (!g_cushion.empty()) CushionDrop(g_cushion.begin(), &g_cushion_drop_pressure);
		return;
	}
	while (!g_cushion.empty() && now - g_cushion.front().t_ms >= g_cushion_age_ms)
		CushionDrop(g_cushion.begin(), &g_cushion_drop_age);
}

// Guarda el bloque si cabe; false = el que llama lo suelta a Windows (bajo g_mutex).
bool CushionPut(Block* b, bool vm) {
	if (!g_cushion_on || b->size > g_cushion_cap || !FullyCommitted(*b) || RamPressure()) return false;
	while (!g_cushion.empty() && g_cushion_bytes.load(std::memory_order_relaxed) + b->size > g_cushion_cap)
		CushionDrop(g_cushion.begin(), &g_cushion_drop_age); // (el mas viejo deja sitio)
	// Copias de la GPU fuera, como al soltar: el siguiente dueno escribe y se sube de nuevo.
	(void)Mem::TryPrepareHostOverwrite(b->addr, b->size);
	g_cushion.push_back(Cushion {std::move(*b), vm, NowMs()});
	g_cushion_bytes.fetch_add(g_cushion.back().b.size, std::memory_order_relaxed);
	g_cushion_n.fetch_add(1, std::memory_order_relaxed);
	return true;
}

// Mismo tamano, misma zona, alineacion compatible; el mas reciente primero (bajo g_mutex).
bool CushionTake(uint64_t size, uint64_t align, bool vm, Block* out) {
	for (auto it = g_cushion.end(); it != g_cushion.begin();) {
		--it;
		if (it->b.size != size || it->vm != vm || (it->b.addr & (align - 1u)) != 0) continue;
		*out = std::move(it->b);
		g_cushion_bytes.fetch_sub(std::min(size, g_cushion_bytes.load(std::memory_order_relaxed)), std::memory_order_relaxed);
		g_cushion_n.fetch_sub(1, std::memory_order_relaxed);
		g_cushion_reused.fetch_add(1, std::memory_order_relaxed);
		g_cushion.erase(it);
		return true;
	}
	return false;
}

// (u1, 11-25) Zona de VirtualMemoryGPU (texturas de streaming): 0x10_0000_0000-0x14_0000_0000, como el asignador del
// juego (0x80d9a0, lStack_f8). Log 20261125-211827-737 (cmd 56): sin esto los bloques de "VirtualMemo" caian en
// 0x0276c00000-0x029e000000 ([memmap]) y toda la ruta de texturas que mira esa zona (VRAM_UNICA en textureCache
// OffloadScan, tex-nativa, anticipada "no legibles 3360", ram16/ram18 "zona VirtualMemo comprometido 0.00") los perdia:
// [vram-unica] 11 trozos (54 MiB) en toda la partida y [ram19] tex en RAM 1.01 GB.
constexpr uint64_t VmZoneBegin = 0x1000000000ull, VmZoneEnd = 0x1400000000ull;
std::atomic<uint64_t> g_vm_blocks {0}, g_vm_outside {0};

// El IAllocator que pide (base 0x8018e0: +0x10 nombre) es VirtualMemoryGPU. Una vez por asignador (cache).
bool IsVirtualMemoryAllocator(uint64_t allocator) {
	static std::unordered_map<uint64_t, bool> cache; // (bajo g_mutex)
	if (allocator == 0) return false;
	if (const auto it = cache.find(allocator); it != cache.end()) return it->second;
	bool     vm       = false;
	uint64_t name_ptr = 0;
	if (Common::GuardedCopy(&name_ptr, reinterpret_cast<const void*>(allocator + 0x10), sizeof(name_ptr)) == 0 &&
	    name_ptr >= 0x10000 && name_ptr < (uint64_t {1} << 47u)) {
		char           name[32] {};
		const uint64_t missing = Common::GuardedCopy(name, reinterpret_cast<const void*>(name_ptr), sizeof(name) - 1);
		name[sizeof(name) - 1 - std::min<uint64_t>(missing, sizeof(name) - 1)] = 0;
		char   packed[32] {};
		size_t n = 0;
		for (const char* p = name; *p != 0 && n + 1 < sizeof(packed); p++)
			if (*p != ' ') packed[n++] = *p;
		vm = std::strncmp(packed, "VirtualMemory", 13) == 0;
		if (vm) {
			std::printf("[port-alloc] asignador \"%s\" (0x%llx): sus bloques en la zona 0x10_0000_0000-0x14_0000_0000 (texturas "
			            "de streaming)\n",
			            name, static_cast<unsigned long long>(allocator));
			std::fflush(stdout);
		}
	}
	cache.emplace(allocator, vm);
	return vm;
}

bool Reserve(uint64_t size, uint64_t align, uint64_t* out, bool vm_zone) {
	if (vm_zone) {
		void* where = reinterpret_cast<void*>(VmZoneBegin);
		if (Mem::KernelReserveVirtualRange(&where, size, 0, align) == 0 && where != nullptr) {
			const auto a = reinterpret_cast<uint64_t>(where);
			if (a >= VmZoneBegin && a + size <= VmZoneEnd && (a & (align - 1u)) == 0) {
				*out = a;
				g_vm_blocks.fetch_add(1, std::memory_order_relaxed);
				return true;
			}
			(void)Mem::KernelMunmap(a, size);
		}
		g_vm_outside.fetch_add(1, std::memory_order_relaxed); // (zona llena: donde haya sitio, como antes)
	}
	for (const uint64_t hint: {uint64_t {0}, Hint}) {
		void* where = reinterpret_cast<void*>(hint);
		if (Mem::KernelReserveVirtualRange(&where, size, 0, align) == 0 && where != nullptr) {
			const auto a = reinterpret_cast<uint64_t>(where);
			// (La tabla de dueno de 1 MiB cubre las direcciones por debajo de 0xfc000 MiB.)
			if (((a + size - 1u) >> 20u) < OwnerMiBCount && (a & (align - 1u)) == 0) {
				*out = a;
				return true;
			}
			(void)Mem::KernelMunmap(a, size);
		}
	}
	return false;
}

void WriteOwner(uint64_t state, uint64_t pool, uint64_t allocator, uint64_t addr, uint64_t size) {
	const uint8_t id = allocator != 0 ? At<uint8_t>(allocator + AllocatorIdAt) : 0;
	if (At<uint8_t>(pool + PIdx) != 0) {
		const uint64_t first = addr >> 20u, last = (addr + size - 1u) >> 20u;
		std::memset(reinterpret_cast<void*>(state + OwnerMiB + first), id, last - first + 1u);
	} else {
		const uint64_t base = At<uint64_t>(state + WinBase), win = At<uint64_t>(state + WinSize);
		if (addr < base || addr + size > base + win) return;
		const uint64_t first = (addr - base) >> 16u, last = (addr + size - 1u - base) >> 16u;
		std::memset(reinterpret_cast<void*>(state + OwnerWin + first), id, last - first + 1u);
	}
}

void AddStat(uint64_t pool, uint64_t at, uint64_t peak_at, uint64_t delta) {
	const uint64_t v = At<uint64_t>(pool + at) + delta;
	At<uint64_t>(pool + at) = v;
	if (At<uint64_t>(pool + peak_at) < v) At<uint64_t>(pool + peak_at) = v;
}
void SubStat(uint64_t pool, uint64_t at, uint64_t delta) {
	const uint64_t v        = At<uint64_t>(pool + at);
	At<uint64_t>(pool + at) = v > delta ? v - delta : 0;
}

// 0x80d9a0 (cerrojo del juego tomado por el que llama; hilo del juego, ABI SysV).
uint64_t KYTY_SYSV_ABI AllocHook(uint64_t allocator, uint64_t info, uint64_t addr, uint64_t size, int32_t type, int32_t align,
                                 int32_t memtype, const uint32_t* flags) {
	(void)memtype;
	(void)flags;
	const uint64_t state = State();
	if (state == 0) {
		g_fails.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	std::lock_guard lock(g_mutex);
	if (info != 0) {
		auto found = g_blocks.find(info);
		if (found == g_blocks.end() || (type != 0 && type != 2) || size == 0 || addr < found->second.addr ||
		    addr + size > found->second.addr + found->second.size) {
			g_fails.fetch_add(1, std::memory_order_relaxed);
			return 0;
		}
		Block&         b    = found->second;
		const uint64_t pool = state + At<uint8_t>(info + DPoolIdx) * PoolStride;
		if (type == 0) {
			// (Lo que ya estuviera comprometido en el tramo se suelta antes: el tramo queda de una pieza.)
			const uint64_t had = DecommitRange(&b, addr, size);
			if (!CommitRange(&b, addr, size)) {
				(void)DecommitRange(&b, addr, size);
				At<uint64_t>(info + DMapped) -= std::min(had, At<uint64_t>(info + DMapped));
				SubStat(pool, PMapped, had);
				PublishBudget(state);
				g_fails.fetch_add(1, std::memory_order_relaxed);
				return 0;
			}
			At<uint64_t>(info + DMapped) += size - had;
			AddStat(pool, PMapped, PMappedPeak, size - had);
		} else {
			const uint64_t done = DecommitRange(&b, addr, size);
			At<uint64_t>(info + DMapped) -= std::min(done, At<uint64_t>(info + DMapped));
			SubStat(pool, PMapped, done);
		}
		PublishBudget(state);
		g_allocs.fetch_add(1, std::memory_order_relaxed);
		return info;
	}
	if (type != 0 && type != 1) {
		g_fails.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	size = (size + Page - 1u) & ~(Page - 1u);
	if (size == 0) {
		g_fails.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	// Alineacion: la del juego, redondeada a 64 KiB y luego a 1 MiB, y como minimo 1 MiB.
	uint64_t a = (static_cast<uint32_t>(align) + 0xffffu) & 0xffff0000u;
	a          = std::max<uint64_t>(MiB, (a + 0xfffffu) & ~uint64_t {0xfffff});
	if ((a & (a - 1u)) != 0) a = MiB;
	uint64_t   va = 0;
	const bool vm = IsVirtualMemoryAllocator(allocator);
	Block      b;
	// (c67) Del colchon: sigue comprometido y reservado (g_reserved / g_mapped no cambian), contenido viejo como en PS5.
	const bool reused = type == 0 && g_cushion_on && CushionTake(size, a, vm, &b);
	if (reused) {
		va = b.addr;
	} else if (!Reserve(size, a, &va, vm)) {
		g_fails.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	b.addr = va;
	b.size = size;
	if (!reused && type == 0 && !CommitRange(&b, va, size)) {
		(void)DecommitRange(&b, va, size);
		(void)Mem::KernelMunmap(va, size);
		g_fails.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	uint8_t* d = NewDescriptor();
	if (d == nullptr) {
		if (reused) {
			ReleaseBlock(&b);
		} else {
			(void)DecommitRange(&b, va, size);
			(void)Mem::KernelMunmap(va, size);
		}
		g_fails.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	const auto     desc   = reinterpret_cast<uint64_t>(d);
	const uint64_t pool   = state + PoolStride; // (0x80d9a0: los bloques nuevos, siempre en el pool 1)
	const uint64_t mapped = type == 0 ? size : 0;
	At<uint8_t>(desc + DPoolIdx)   = At<uint8_t>(pool + PIdx);
	At<uint64_t>(desc + DSize)     = size;
	At<uint64_t>(desc + DAddr)     = va;
	At<uint8_t>(desc + DInUse)     = 1;
	At<uint64_t>(desc + DAlloc)    = allocator;
	At<uint32_t>(desc + DIndex48)  = 0xffffffffu;
	At<uint64_t>(desc + DMapped)   = mapped;
	WriteOwner(state, pool, allocator, va, size);
	// Nodo en la lista de en uso del pool (al final), como 0x80dfc9..
	At<uint64_t>(desc + DNode)               = desc;
	At<uint64_t>(desc + DNext)               = pool;
	At<uint64_t>(desc + DPrev)               = At<uint64_t>(pool + PTail);
	At<uint64_t>(pool + PTail)               = desc + DNode;
	At<uint64_t>(At<uint64_t>(desc + DPrev) + 0x10) = desc + DNode;
	At<uint64_t>(desc + DPool)               = pool;
	At<int32_t>(pool + PBlocks) += 1;
	AddStat(pool, PSpan, PSpanPeak, size);
	if (mapped != 0) AddStat(pool, PMapped, PMappedPeak, mapped);
	PublishBudget(state);
	if (!reused) g_reserved.fetch_add(size, std::memory_order_relaxed);
	g_blocks.emplace(desc, std::move(b));
	g_allocs.fetch_add(1, std::memory_order_relaxed);
	return desc;
}

void TakeGameLock() {
	auto& lock = *reinterpret_cast<std::atomic<int64_t>*>(g_lock);
	for (uint32_t spins = 0;;) {
		int64_t expected = 0;
		if (lock.load(std::memory_order_relaxed) == 0 &&
		    lock.compare_exchange_weak(expected, -1, std::memory_order_acquire, std::memory_order_relaxed))
			return;
		if (++spins >= 10000) {
			spins = 0;
			std::this_thread::sleep_for(std::chrono::microseconds(1));
		} else {
			std::this_thread::yield();
		}
	}
}
void ReleaseGameLock() {
	reinterpret_cast<std::atomic<int64_t>*>(g_lock)->store(0, std::memory_order_release);
}

// 0x80f550 (hilo del juego, ABI SysV).
void KYTY_SYSV_ABI FreeHook(uint64_t desc) {
	{
		std::unique_lock lock(g_mutex);
		if (g_blocks.find(desc) == g_blocks.end()) {
			lock.unlock();
			g_original_free(desc); // (no es de aqui)
			return;
		}
	}
	TakeGameLock();
	{
		std::lock_guard lock(g_mutex);
		auto            found = g_blocks.find(desc);
		if (found != g_blocks.end()) {
			const uint64_t state = State();
			const uint64_t pool  = state + At<uint8_t>(desc + DPoolIdx) * PoolStride;
			At<uint64_t>(g_state_global - OpsFromState) += 1; // ([0x2ee4d50], como 0x80f550)
			// Fuera de la lista de en uso.
			const uint64_t prev = At<uint64_t>(desc + DPrev), next = At<uint64_t>(desc + DNext);
			At<uint64_t>(prev + 0x10) = next;
			At<uint64_t>(next + 0x08) = prev;
			At<int32_t>(pool + PBlocks) -= 1;
			SubStat(pool, PSpan, At<uint64_t>(desc + DSize));
			SubStat(pool, PMapped, At<uint64_t>(desc + DMapped));
			Block& b = found->second;
			// (c67) Entero comprometido y con sitio: al colchon; si no, a Windows (ReleaseBlock).
			if (g_cushion_on) CushionTrim(NowMs());
			if (!CushionPut(&b, IsVirtualMemoryAllocator(At<uint64_t>(desc + DAlloc)))) ReleaseBlock(&b);
			g_blocks.erase(found);
			PublishBudget(state);
			std::memset(reinterpret_cast<void*>(desc), 0, DescSize);
			g_free_descs.push_back(reinterpret_cast<uint8_t*>(desc));
			g_frees.fetch_add(1, std::memory_order_relaxed);
		}
	}
	ReleaseGameLock();
}

double Mib(uint64_t b) {
	return static_cast<double>(b) / 1048576.0;
}

void ReportThread() {
	LocalPlatform::SetThreadName("Kyty.PortAlloc");
	uint64_t last_allocs = 0, last_frees = 0, last_reused = 0;
	for (;;) {
		for (int i = 0; i < 50 && !g_stop.load(std::memory_order_relaxed); i++) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			if (g_cushion_on && g_cushion_n.load(std::memory_order_relaxed) != 0) {
				std::lock_guard lock(g_mutex); // (solo memoria del emulador: esos bloques ya no son del juego)
				CushionTrim(NowMs());
			}
		}
		if (g_stop.load(std::memory_order_relaxed)) break;
		uint64_t commit = 0, ram = 0;
#if defined(_WIN32)
		PROCESS_MEMORY_COUNTERS_EX pmc {};
		pmc.cb = sizeof(pmc);
		if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)) != 0) {
			commit = pmc.PrivateUsage;
			ram    = pmc.WorkingSetSize;
		}
#endif
		const uint64_t allocs = g_allocs.load(std::memory_order_relaxed), frees = g_frees.load(std::memory_order_relaxed);
		size_t         blocks = 0;
		{
			std::lock_guard lock(g_mutex);
			blocks = g_blocks.size();
		}
		std::printf("[port-alloc] 5 s: reservado %.0f MiB, comprometido (proceso) %.0f MiB, en RAM %.0f MiB, en uso %.0f MiB "
		            "(pico %.0f), %llu bloques, %.1f asig/s, %.1f lib/s, fallos %llu, devuelto a Windows %.0f MiB, claves "
		            "troceadas %llu; bloques de VirtualMemo en su zona %llu, fuera (zona llena) %llu; colchon %.0f MiB (bloques %llu), "
		            "reutilizados %.1f/s, soltados por presion %.0f MiB, por edad %.0f MiB\n",
		            Mib(g_reserved.load(std::memory_order_relaxed)), Mib(commit), Mib(ram),
		            Mib(g_mapped.load(std::memory_order_relaxed)), Mib(g_peak.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(blocks), static_cast<double>(allocs - last_allocs) / 5.0,
		            static_cast<double>(frees - last_frees) / 5.0,
		            static_cast<unsigned long long>(g_fails.load(std::memory_order_relaxed)),
		            Mib(g_decommitted.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_key_splits.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_vm_blocks.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_vm_outside.load(std::memory_order_relaxed)),
		            Mib(g_cushion_bytes.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_cushion_n.load(std::memory_order_relaxed)),
		            static_cast<double>(g_cushion_reused.load(std::memory_order_relaxed) - last_reused) / 5.0,
		            Mib(g_cushion_drop_pressure.load(std::memory_order_relaxed)),
		            Mib(g_cushion_drop_age.load(std::memory_order_relaxed)));
		std::fflush(stdout);
		last_allocs = allocs;
		last_frees  = frees;
		last_reused = g_cushion_reused.load(std::memory_order_relaxed);
	}
	g_running.store(false, std::memory_order_release);
}

#if defined(_WIN32)
bool WriteCode(uint64_t site, const uint8_t* bytes, size_t size) {
	DWORD old = 0;
	if (VirtualProtect(reinterpret_cast<void*>(site), size, PAGE_EXECUTE_READWRITE, &old) == 0) return false;
	std::memcpy(reinterpret_cast<void*>(site), bytes, size);
	DWORD ignored = 0;
	(void)VirtualProtect(reinterpret_cast<void*>(site), size, old, &ignored);
	(void)FlushInstructionCache(GetCurrentProcess(), reinterpret_cast<void*>(site), size);
	return true;
}
#endif

} // namespace

bool Requested() {
	static const bool on = [] {
		// (11-26 verde) encendido por defecto: validado en pruebas 62-69; KYTY_PORT_ALLOC=0 lo apaga.
		const char* v = std::getenv("KYTY_PORT_ALLOC");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

void Install(Program* program) {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(_WIN32)
	if (!Requested() || g_alloc_site != 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	// El pool de PS5 del juego ya no se usa: que Init (0x80c200) lo deje en lo minimo (DemonsSoulsAllocClamp lo lee despues).
	if (std::getenv("KYTY_MEM_DIRECTA_MB") == nullptr || std::getenv("KYTY_MEM_DIRECTA_MB")[0] == '\0') {
		(void)_putenv_s("KYTY_MEM_DIRECTA_MB", "64");
	}
	const auto               alloc_site = GuestCode::FindUnique(*program, GuestCode::Pattern(RequestPages));
	const auto               free_site  = GuestCode::FindUnique(*program, GuestCode::Pattern(PF::FreeBlock));
	const GuestCode::Pattern body(PF::FreeBlockBody);
	if (!alloc_site || !free_site || !Libs::Graphics::HostMemoryRangeIsReadable(*free_site + PF::FreeBlockBodyAt, body.Size()) ||
	    !body.Matches(reinterpret_cast<const uint8_t*>(*free_site + PF::FreeBlockBodyAt))) {
		std::printf("[port-alloc] no se encontro el asignador de paginas (01.007.000): apagado\n");
		std::fflush(stdout);
		return;
	}
	int32_t stats_disp = 0, lock_disp = 0;
	std::memcpy(&stats_disp, reinterpret_cast<const void*>(*free_site + PF::StatsOperandAt), sizeof(stats_disp));
	std::memcpy(&lock_disp, reinterpret_cast<const void*>(*free_site + PF::LockOperandAt), sizeof(lock_disp));
	g_state_global = *free_site + PF::StatsOperandEnd + static_cast<int64_t>(stats_disp);
	g_lock         = *free_site + PF::LockOperandEnd + static_cast<int64_t>(lock_disp);

	const char* budget = std::getenv("KYTY_PORT_ALLOC_PRESUPUESTO_MB");
	const auto  mb     = budget != nullptr ? std::strtoull(budget, nullptr, 10) : 0ull;
	g_budget           = (mb != 0 ? static_cast<uint64_t>(mb) : uint64_t {12288}) << 20u;

	// (c67) Colchon (KYTY_PORT_ALLOC_COLCHON=1; tope KYTY_PORT_ALLOC_COLCHON_MB = 256; edad KYTY_PORT_ALLOC_COLCHON_S = 10).
	const char* cushion = std::getenv("KYTY_PORT_ALLOC_COLCHON");
	// (11-26 verde) encendido por defecto: validado en pruebas 62-69; KYTY_PORT_ALLOC_COLCHON=0 lo apaga.
	g_cushion_on        = cushion != nullptr && cushion[0] == '1'; // (11-26 03:10) APAGADO por defecto: sospechoso de parpadeos/fallos graficos con el umbral de 256
	const char* cap_mb  = std::getenv("KYTY_PORT_ALLOC_COLCHON_MB");
	const auto  cap     = cap_mb != nullptr ? std::strtoull(cap_mb, nullptr, 10) : 0ull;
	g_cushion_cap       = (cap != 0 ? static_cast<uint64_t>(cap) : uint64_t {256}) << 20u;
	const char* age_s   = std::getenv("KYTY_PORT_ALLOC_COLCHON_S");
	const auto  age     = age_s != nullptr ? std::strtoull(age_s, nullptr, 10) : 0ull;
	g_cushion_age_ms    = (age != 0 ? static_cast<uint64_t>(age) : uint64_t {10}) * 1000u;

	// Trampolin de 0x80f550 para los descriptores ajenos: prologo (13 bytes) + jmp [rip+0] a site+13.
	auto* tramp = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	if (tramp == nullptr) return;
	std::memcpy(tramp, reinterpret_cast<const void*>(*free_site), FreePrologue);
	const uint8_t  jmp_back[6] = {0xff, 0x25, 0x00, 0x00, 0x00, 0x00};
	const uint64_t back        = *free_site + FreePrologue;
	std::memcpy(tramp + FreePrologue, jmp_back, sizeof(jmp_back));
	std::memcpy(tramp + FreePrologue + sizeof(jmp_back), &back, sizeof(back));
	DWORD old = 0;
	if (VirtualProtect(tramp, 4096, PAGE_EXECUTE_READ, &old) == 0) {
		VirtualFree(tramp, 0, MEM_RELEASE);
		return;
	}
	g_original_free = reinterpret_cast<FreeFn>(tramp);

	// Entradas: mov rax, gancho; jmp rax (rax no lleva argumento: ninguna de las dos es variadica).
	uint8_t        alloc_patch[JumpSize]    = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0};
	uint8_t        free_patch[FreePrologue] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0, 0x90};
	const uint64_t alloc_hook               = reinterpret_cast<uint64_t>(&AllocHook);
	const uint64_t free_hook                = reinterpret_cast<uint64_t>(&FreeHook);
	std::memcpy(alloc_patch + 2, &alloc_hook, sizeof(alloc_hook));
	std::memcpy(free_patch + 2, &free_hook, sizeof(free_hook));
	if (!WriteCode(*free_site, free_patch, sizeof(free_patch))) {
		std::printf("[port-alloc] no se pudo escribir la entrada de la liberacion: apagado\n");
		std::fflush(stdout);
		g_original_free = nullptr;
		return;
	}
	if (!WriteCode(*alloc_site, alloc_patch, sizeof(alloc_patch))) {
		// (Sin el alta nativa no hay descriptores de aqui: FreeHook pasa todo al original.)
		std::printf("[port-alloc] no se pudo escribir la entrada de RequestVirtualPages: apagado\n");
		std::fflush(stdout);
		return;
	}
	g_alloc_site = *alloc_site;
	g_free_site  = *free_site;
	g_active.store(true, std::memory_order_release);

	g_stop.store(false, std::memory_order_relaxed);
	g_running.store(true, std::memory_order_release);
	std::thread(ReportThread).detach();
	std::printf("[port-alloc] asignador de paginas del juego sustituido por memoria nativa de Windows: RequestVirtualPages "
	            "eboot+0x%llx, Free eboot+0x%llx; estado [eboot+0x%llx], cerrojo [eboot+0x%llx]; pool de PS5 %s MiB; "
	            "presupuesto anunciado %llu MiB; commit-real %s; colchon %s (tope %llu MiB, %llu s)\n",
	            static_cast<unsigned long long>(g_alloc_site - program->base_vaddr),
	            static_cast<unsigned long long>(g_free_site - program->base_vaddr),
	            static_cast<unsigned long long>(g_state_global - program->base_vaddr),
	            static_cast<unsigned long long>(g_lock - program->base_vaddr), std::getenv("KYTY_MEM_DIRECTA_MB"),
	            static_cast<unsigned long long>(g_budget >> 20u), NativeLazyCommit::CommitRealRequested() ? "si" : "NO",
	            g_cushion_on ? "si" : "no", static_cast<unsigned long long>(g_cushion_cap >> 20u),
	            static_cast<unsigned long long>(g_cushion_age_ms / 1000u));
	std::fflush(stdout);
#else
	(void)program;
#endif
}

void Clear() {
	g_stop.store(true, std::memory_order_relaxed);
	for (int i = 0; i < 300 && g_running.load(std::memory_order_acquire); i++)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	// (El codigo del juego se descarga con el programa: los ganchos se van con el.)
	g_alloc_site = 0;
	g_free_site  = 0;
	g_active.store(false, std::memory_order_release);
}

} // namespace Loader::DemonsSoulsPortAlloc
