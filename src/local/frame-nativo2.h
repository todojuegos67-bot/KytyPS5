#pragma once
// (FRAME-NATIVO2) KYTY_FRAME_NATIVO2=1 (apagado por defecto). Paso 1 nativo: los 4 arenas "DefaultTagg" (el padre de
// los TaggedFrameAllocator, 120 MiB cada uno, fijos: 0x200b80000, 0x261700000, 0x375800000, 0x3fad00000) se importan
// enteros como VkBuffer (VK_EXT_external_memory_host) mientras siguen RW (sin vigilar: el import de memoria READONLY lo
// rechaza NVIDIA, log 20261124-091857-518). Desde ahi:
//  - toda lectura de buffer (V#, no texel, no escrita por la GPU) que cae entera dentro de un arena se sirve de ese
//    VkBuffer: ni se registra en la cache, ni se vigila (sin proteccion), ni se resube;
//  - las entradas BDA de las paginas del arena apuntan siempre a ese VkBuffer (WriteBdaEntries) y EnsureBdaReadable
//    no crea buffers vigilados sobre ellas.
//
// Ghidra (eboot 01.007.000): TaggedFrameAllocator::Init 0x81d250(this, padre, num_buffers) guarda [+0x88]=padre,
// [+0x94]=num_buffers; NextFrame 0x81d610 hace idx=([+0x90]+1)%[+0x94] y devuelve al padre (vtable+0x100, clave
// tag[+0x30]<<32|idx) los trozos de ese indice. 15 instancias: num_buffers 1 (8), 2 (3), 3 (2), 5 (1), 6 (1); los
// padres son los asignadores por defecto 0x806bd0 / 0x8069d0 (compartidos). Un trozo liberado por un asignador de 2
// buffers lo puede coger al momento uno de 1: NINGUN arena es "solo de 2 buffers"; la coherencia no se puede elegir
// por arena. En PS5 la GPU tambien lee esta memoria en vivo: el juego solo reutiliza tras esperar sus etiquetas de fin
// de GPU. Por eso, con un arena importado, las etiquetas de fin (EOP/RELEASE_MEM a memoria) se escriben al COMPLETAR
// la GPU de verdad (P7 C5, DeferLabelWrite) y no al traducir: el juego no reescribe antes de que Vulkan lo haya leido.
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>
#include <x86intrin.h>

#include "arena-dueno.h" // (v11, =4: la tabla de duenos de los bloques, encendida por dentro)

namespace FrameNativo2 {

// (FRAME-NATIVO2 v5) KYTY_FRAME_NATIVO2=1: COPIA. Log 20261124-102634-504: con el modo vivo (C5) el hilo de render
// espera a la GPU ("[fps] ... cpu 2, idle 26-34 ms" frente a "idle 5.8" antes del import; "[p7] C5 ... ~52 ms/frame").
// Ghidra/objdump (eboot 01.007.000, VA reales = las de Ghidra - 0xbdb0): Init 0x8114a0 guarda [+0x88]=padre,
// [+0x94]=num_buffers; NextFrame (0x811c27..0x811c51) hace idx=([+0x90]+1)%[+0x94] y llama al padre (vtable+0x100)
// sin esperar nada: la unica proteccion contra la reescritura es la etiqueta de fin que el juego espera. Sin C5 esa
// etiqueta llega al traducir, asi que la GPU no puede leer el arena en vivo (ni copiarlo en su linea de tiempo: la
// copia correria igual despues de la etiqueta). En COPIA cada lectura V# del arena se copia al traducir (el mismo
// instante en que el camino de siempre subia sus bytes) a la anilla de subida de 64 MiB: ~98 MB/s, ~3 MB/frame, sin
// proteger, sin registrar, sin fallos y con las etiquetas al traducir (sin C5). BDA del arena: camino de la cache, su
// rango marcado como modificado por la CPU en cada uso (las paginas ya no se protegen).
// (v6) COPIA ya no importa ni excluye el arena (g_imported = solo "direccion conocida"): sin proteccion el juego
// reescribia trozos que la GPU habia escrito sin avisar al tracker (cuelgue S# + glitch, log 20261124-105755-202).
// KYTY_FRAME_NATIVO2=2: VIVO (v4: lectura en vivo del buffer importado + C5), para comparar.
[[nodiscard]] inline int Mode() noexcept {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_FRAME_NATIVO2");
		return v == nullptr ? 0 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : v[0] == '3' ? 3 : v[0] == '4' ? 4 : v[0] == '5' ? 5 : v[0] == '6' ? 6 : 0;
	}();
	static const bool said = [] { // (v13) al arrancar: el valor leido y el modo que queda
		const char* v = std::getenv("KYTY_FRAME_NATIVO2");
		std::printf("[frame-nativo2] KYTY_FRAME_NATIVO2=%s -> modo %d%s\n", v == nullptr ? "(no puesta)" : v, mode,
		            mode == 6 ? " (v13 NATIVO: ganchos NextFrame/FreeTag/AllocTag e import del arena)" : "");
		std::fflush(stdout);
		return true;
	}();
	(void)said;
	return mode;
}
[[nodiscard]] inline bool Wanted() noexcept { return Mode() != 0; }
[[nodiscard]] inline bool Live() noexcept { return Mode() == 2; }
// (FRAME-NATIVO2 v8) KYTY_FRAME_NATIVO2=3: COPIA (v7) + arena SIN proteger (excluido como v6). Lo que la GPU escribe en
// el arena (usos 'escritos' de GetBuffer) se apunta con el fotograma del juego (g_frames, gancho NextFrame 0x811860); en
// el hilo de la GPU, al ver un NextFrame del bucle nuevo, esos rangos de fotogramas anteriores se DESCARTAN (pasan a
// la CPU: DiscardGpuWrites) antes de traducir nada del fotograma nuevo: NextFrame es el unico punto de reutilizacion.
[[nodiscard]] inline bool NoProt() noexcept { return Mode() == 3; }
// (FRAME-NATIVO2 v11) KYTY_FRAME_NATIVO2=4: SOLO LECTURA (ver el final de este archivo).
// (FRAME-NATIVO2 v12) KYTY_FRAME_NATIVO2=5: SOLO LECTURA POR BLOQUES (=4 con la epoca de cada bloque; ver el final).
// (FRAME-NATIVO2 v13) KYTY_FRAME_NATIVO2=6: NATIVO (arena importado + cuarentena de FreeTag; ver el final).
[[nodiscard]] inline bool SoloLectura() noexcept { return Mode() == 4 || Mode() == 5 || Mode() == 6; }
[[nodiscard]] inline bool V12() noexcept { return Mode() == 5; }
[[nodiscard]] inline bool Nativo() noexcept { return Mode() == 6; }
// (v13) Gancho FreeTag puesto (demonsSoulsCopy.cpp): sin el no hay cuarentena y ningun bloque sale de la vigilancia.
inline std::atomic<bool> g_freetag_hooked {false};
[[nodiscard]] inline bool NativoReady() noexcept { return g_freetag_hooked.load(std::memory_order_acquire); }
// (v13) Fotogramas del juego ya traducidos enteros (graphicsRun.cpp: FrameBoundary consumido, o Done sin tuberia).
inline std::atomic<uint64_t> g_guest_consumed {0};
inline void HoldFreed(uint64_t parent, uint64_t chunk, uint64_t base, uint64_t block, int slot, uint64_t held0, uint64_t held1) noexcept; // (v13)
inline std::atomic<uint64_t> g_inval_frames {0}, g_inval_ranges {0}, g_inval_bytes {0}, g_gpu_writes {0};

constexpr uint64_t kArenaSize = 0x7800000; // 120 MiB (each DefaultTagg arena seen so far)
constexpr int      kArenas    = 4;
// (FRAME-NATIVO2 v3) The arenas' guest addresses are NOT fixed: log 20261124-100322-504 has 0x200b80000, 0x261700000,
// 0x36f800000, 0x3b1900000 (v2 assumed 0x375800000 / 0x3fad00000). Taken at run time from the game's own allocator
// list ([memmap], Loader::DemonsSoulsMemMap::RangesOf("DefaultTagg"), ranges of kArenaSize); 0 = slot not found yet.
// Written once per slot (render thread, before g_excluded/g_imported of that slot), never changed after.
inline std::atomic<uint64_t> g_base[kArenas] {};
[[nodiscard]] inline uint64_t Base(int i) noexcept { return g_base[i].load(std::memory_order_acquire); }

// Set (release) once the arena is imported; read lock-free by the lookup.
inline std::atomic<bool>     g_imported[kArenas] {};
// (FRAME-NATIVO2 v2) Set before the arena is made read-write for its import, and never cleared once imported: the
// trackers' (and RAIZ's) protections of its pages are dropped (ProtectTransient, memoryAddressSpace.inc). The driver
// refuses a READONLY range at the import ([unificada] probe), so the import runs right after the unprotect.
inline std::atomic<bool>     g_excluded[kArenas] {};
inline std::atomic<uint64_t> g_skipped {0}; // protections dropped on excluded arenas
inline std::atomic<int>      g_imported_count {0};
// (v3) Why a try did not import, one counter per reason (the [frame-nativo2] line).
enum Reason { RNoBase, RNoExtension, RNotMapped, RRestore, RNoBacking, RQuery, RNotRw, RVulkan, RReasons };
inline std::atomic<uint64_t> g_reason[RReasons] {};
inline std::atomic<uint64_t> g_hits {0}, g_hit_bytes {0}, g_fails {0}, g_tries {0}, g_bda_skips {0};

// Some arena imported in VIVO mode: labels at GPU completion (p7-compat.h LabelsAtCompletion), BDA to the arena.
[[nodiscard]] inline bool Active() noexcept { return Live() && g_imported_count.load(std::memory_order_relaxed) > 0; }
// (v5) COPIA: reads copied at translation (the [frame-nativo2] line).
inline std::atomic<uint64_t> g_copies {0}, g_copy_bytes {0}, g_copy_fallbacks {0};

// Index of the imported arena holding all of [vaddr, vaddr + size), or -1.
[[nodiscard]] inline int Find(uint64_t vaddr, uint64_t size) noexcept {
	for (int i = 0; i < kArenas; ++i) {
		const uint64_t b = Base(i);
		if (b != 0 && vaddr >= b && size <= kArenaSize && vaddr - b <= kArenaSize - size)
			return g_imported[i].load(std::memory_order_acquire) ? i : -1;
	}
	return -1;
}

// Some arena excluded (one acquire load per arena; all false while off).
[[nodiscard]] inline bool AnyExcluded() noexcept {
	for (int i = 0; i < kArenas; ++i)
		if (g_excluded[i].load(std::memory_order_acquire)) return true;
	return false;
}

// Any arena (imported or not) touching [vaddr, vaddr + size).
[[nodiscard]] inline int Touching(uint64_t vaddr, uint64_t size) noexcept {
	for (int i = 0; i < kArenas; ++i)
		if (const uint64_t b = Base(i); b != 0 && vaddr < b + kArenaSize && b < vaddr + size) return i;
	return -1;
}

// (FRAME-NATIVO2 v7) Log 20261124-111746-716 (COPIA): ~2370 copias/frame, ~55 MB/frame (2.3 GB/s) copiados al traducir y
// el arena sigue protegido (DefaultTagg 18.5k/s): frame 23-24 ms frente a 10 ms sin FRAME-NATIVO2. La mayoria son el
// MISMO rango (constantes por pase) leido por muchos draws. Ghidra (archivo 0x81d610 = juego 0x811860, NextFrame):
// idx=([+0x90]+1)%[+0x94]; padre->vtable[+0x100](tag<<32|idx) devuelve los trozos de ese indice: es el UNICO punto
// donde memoria del arena vuelve a quedar libre para reescribirse. Lo llaman el bucle de fotograma (call en 0xa988e6,
// vuelve a 0xa988eb) y 0xbf0b31. Regla: entre dos llamadas a NextFrame (de cualquier instancia) ningun byte ya entregado
// a la GPU se reescribe; asi que una copia de [vaddr,size) hecha en la epoca E y en el mismo tick de la anilla vale
// para todas las lecturas siguientes mientras la epoca siga siendo E. El gancho (13 bytes en 0x811860, loader/
// demonsSoulsCopy.cpp) sube g_epoch ANTES de que el juego libere nada. Sin gancho: sin reutilizacion (como v6).
inline std::atomic<uint64_t> g_epoch {0};          // NextFrame calls (every instance)
inline std::atomic<uint64_t> g_frames {0};         // NextFrame calls from the frame loop (return 0xa988eb)
inline std::atomic<bool>     g_hooked {false};
inline std::atomic<uint64_t> g_frame_loop_ret {0}; // eboot base + 0xa988eb
inline std::atomic<uint64_t> g_reuses {0}, g_reuse_bytes {0}, g_copy_ns {0};
inline void OnNextFrame(uint64_t ret) noexcept {
	g_epoch.fetch_add(1, std::memory_order_seq_cst);
	if (ret == g_frame_loop_ret.load(std::memory_order_relaxed)) g_frames.fetch_add(1, std::memory_order_relaxed);
}

// (FRAME-NATIVO2 v9, =3) Ghidra: el padre de TaggedFrameAllocator ([+0x88], puesto en Init 0x8114a0) es el asignador
// global de 0x730 bytes (getter archivo 0x8069d0 -> global archivo 0x2eedbe8). Su vtable+0x100 es FreeTag (juego
// 0x7fe540 = archivo 0x80a2f0): busca la ranura s con [+0x528+s*8]==clave (bit s de [+0x728]), recorre los montones
// (lista [+0x518]) y sus trozos (lista monton+0x10); en cada trozo c: base [c+0x10], mascara de 120 bloques de
// [padre+0x88] (1 MiB) en [c+0x28+s*0x10] (128 bits, alto bzhi 0x38); [c+0x18] &= ~mascara; mascara = 0. Devuelve
// bloques*[+0x88]. Cerrojo: *[+0x80] (0 -> -1 con cmpxchg). Esto apunta ANTES de liberar los rangos EXACTOS (bloques
// contiguos unidos); el hilo de la GPU los descarta (DiscardGpuWrites + CPU-modified) antes de traducir nada que los lea.
inline std::mutex                                  g_freed_mutex;
inline std::vector<std::pair<uint64_t, uint64_t>> g_freed;
inline std::atomic<uint64_t> g_free_calls {0}, g_free_found {0}, g_freed_ranges {0}, g_freed_bytes {0};
inline std::atomic<uint64_t> g_drained_ranges {0}, g_drained_bytes {0}, g_drained_gpu {0};
// (v10, =3) Log 20261124-121448-571: v9 marcaba CPU-modificado TODO lo liberado (1976 MB/s en bloques de 1 MiB) y cada
// uso por la cache lo volvia a subir: 75-85 ms/frame. Ahora al liberar solo se suelta lo que la GPU tenia (sin
// descargar: el juego lo sobrescribe) y la epoca de copia sube; se marca solo lo que se usa. Cifras del [frame-nativo2 v10].
inline std::atomic<uint64_t> g_marked_bytes {0}, g_reupload_bytes {0}, g_dropped_gpu_bytes {0}, g_bda_memo {0};
inline std::atomic<int>      g_vtable_ok {-1}; // 1: padre->vtable[+0x100] == juego 0x7fe540 (comprobado en NextFrame)
inline void OnFreedRo(uint64_t lo, uint64_t size) noexcept; // (v11, =4: abajo)
inline void CollectFreed(uint64_t parent, uint64_t key) noexcept {
	g_free_calls.fetch_add(1, std::memory_order_relaxed);
	const auto rd   = [](uint64_t a) { return *reinterpret_cast<const volatile uint64_t*>(a); };
	auto*      lock = reinterpret_cast<std::atomic<int64_t>*>(rd(parent + 0x80));
	if (lock != nullptr) {
		for (int64_t z = 0; !lock->compare_exchange_weak(z, -1, std::memory_order_acquire); z = 0) _mm_pause();
	}
	std::vector<std::pair<uint64_t, uint64_t>> out;
	const uint64_t used = rd(parent + 0x728);
	int            slot = -1;
	for (int s = 0; s < 64; ++s)
		if (((used >> s) & 1u) != 0 && rd(parent + 0x528 + static_cast<uint64_t>(s) * 8) == key) {
			slot = s;
			break;
		}
	const uint64_t block = rd(parent + 0x88);
	if (slot >= 0 && block != 0) {
		const uint64_t heads = parent + 0x518;
		int            guard = 0;
		for (uint64_t h = rd(heads); h != heads && h != 0 && guard < 4096; h = rd(h), ++guard) {
			const uint64_t chunks = h + 0x10;
			for (uint64_t c = rd(chunks); c != chunks && c != 0 && guard < 4096; c = rd(c), ++guard) {
				const uint64_t base = rd(c + 0x10);
				uint64_t       m[2] = {rd(c + 0x28 + static_cast<uint64_t>(slot) * 0x10),
				                       rd(c + 0x30 + static_cast<uint64_t>(slot) * 0x10) & ((uint64_t {1} << 0x38) - 1)};
				// (v13, =6) Bloques sin vigilar (leidos en vivo por la GPU desde el arena importado): se quitan de la mascara
				// de la clave ANTES de que FreeTag la lea (siguen ocupados en [c+0x18]: AllocTag no los da) y quedan en
				// cuarentena hasta que la GPU termine con ellos (HoldFreed / FrameNativo2DrainHeld).
				if (Nativo() && base != 0) HoldFreed(parent, c, base, block, slot, m[0], m[1]);
				if (Nativo() && base != 0) {
					m[0] = rd(c + 0x28 + static_cast<uint64_t>(slot) * 0x10);
					m[1] = rd(c + 0x30 + static_cast<uint64_t>(slot) * 0x10) & ((uint64_t {1} << 0x38) - 1);
				}
				for (uint64_t i = 0; i < 120 && base != 0;) {
					if (((m[i >> 6] >> (i & 63)) & 1u) == 0) {
						++i;
						continue;
					}
					uint64_t j = i;
					while (j < 120 && ((m[j >> 6] >> (j & 63)) & 1u) != 0) ++j;
					out.emplace_back(base + i * block, (j - i) * block);
					i = j;
				}
			}
		}
	}
	if (lock != nullptr) lock->store(0, std::memory_order_release);
	if (SoloLectura())
		for (const auto& r: out) OnFreedRo(r.first, r.second); // (v11) sin vigilar -> libre sin vigilar, epoca
	if (out.empty()) return;
	g_free_found.fetch_add(1, std::memory_order_relaxed);
	uint64_t bytes = 0;
	for (const auto& r: out) bytes += r.second;
	g_freed_ranges.fetch_add(out.size(), std::memory_order_relaxed);
	g_freed_bytes.fetch_add(bytes, std::memory_order_relaxed);
	const std::lock_guard l(g_freed_mutex);
	g_freed.insert(g_freed.end(), out.begin(), out.end());
}
inline void TakeFreed(std::vector<std::pair<uint64_t, uint64_t>>& out) {
	const std::lock_guard l(g_freed_mutex);
	out.swap(g_freed);
	g_freed.clear();
}

// (FRAME-NATIVO2 v11) KYTY_FRAME_NATIVO2=4: SOLO LECTURA. Medidor ARENA-DUENO (log 20261124-192419-628): la GPU escribe
// SOLO en los bloques de la instancia tag 41 (Init desde eboot+0xa8fa9d: miembro en +0 del objeto de render cuyo
// [+0x156fc] sube el bucle de fotograma antes de NextFrame 0xa988e6); en las de tag 30 (Init desde eboot+0xaa2152,
// global estatica juego 0x2f65470, 2 buferes, 314 MiB) y tag 28 (eboot+0xaa20cb, global juego 0x2f64b20, 5 buferes)
// solo lee. AllocTag (0x7fe350) reparte bloques de 1 MiB por clave tag<<32|idx: un bloque es de una sola instancia.
// Asi: los bloques que AllocTag da a una instancia de solo lectura salen de la vigilancia (sus protecciones se
// descartan en ProtectTransient y se dejan RW con RestoreGuestWritable, como v8 pero SOLO esos bloques); la GPU los lee
// por la copia por epoca de v7 (FrameNativo2Obtain) y la BDA por rangos exactos (marca de cada uso). En FreeTag pasan a
// "libre sin vigilar" y se invalidan las copias (epoca); si AllocTag los da despues a una instancia vigilada vuelven
// a la vigilancia (CPU-modificado entero en el hilo de la GPU). Red de seguridad: una escritura de la GPU en un bloque
// sin vigilar se apunta y la instancia entera vuelve a la vigilancia normal.
// Lista configurable: KYTY_FRAME_NATIVO2_RO = "0xaa2152,0xaa20cb" (llamadores de Init, desplazamiento en el eboot) y/o
// "t30,t28" (tags). Por defecto los dos llamadores.
constexpr int      kRoSlots = 4096;
constexpr int      kRoShift = 19; // granulos de 512 KiB (bases de arena alineadas a 512 KiB, bloques base + i MiB)
enum RoState : uint8_t { RoWatched = 0, RoUnwatched = 1, RoFreeUnwatched = 2 };
inline std::atomic<uint64_t> g_ro_key[kRoSlots] {};
inline std::atomic<uint8_t>  g_ro_state[kRoSlots] {};
// (v12, =5) Epoca POR BLOQUE: g_ro_gen[s] sube en cada cambio de estado del granulo s (AllocTag lo da a una instancia,
// FreeTag lo libera, vuelve a la vigilancia, red de seguridad). Una copia/subida vale mientras la suma de las epocas de
// sus granulos no cambie (solo crecen: la suma cambia si y solo si alguna cambia).
inline std::atomic<uint32_t> g_ro_gen[kRoSlots] {};
inline std::atomic<uint64_t> g_ro_lo {UINT64_MAX}, g_ro_hi {0};
inline std::atomic<int64_t>  g_ro_granules {0}; // granulos sin vigilar ahora (estado 1 o 2)
inline std::atomic<bool>     g_ro_bad[ArenaDueno::kInstances] {}; // instancia devuelta a la vigilancia (red de seguridad)
inline std::atomic<uint64_t> g_ro_unexpected {0}, g_ro_allocs {0}, g_ro_alloc_bytes {0}, g_ro_restore_fail {0};
inline std::atomic<uint64_t> g_ro_rewatch {0}, g_ro_rewatch_bytes {0}, g_ro_freed {0};
inline std::mutex                                  g_rewatch_mutex;
inline std::vector<std::pair<uint64_t, uint64_t>> g_rewatch; // rangos que vuelven a la vigilancia (hilo de la GPU)
// (v11b) Log 20261124-194322-958: tablas que no se evaluan ("a descriptor source did not evaluate") y variantes nuevas
// sin fin. Los bloques que AllocTag da a una instancia de solo lectura podian conservar el estado "escrito por la GPU"
// de su dueno anterior (FreeTag de una instancia vigilada): el lector limpio de descriptores (TryReadGpuCleanBackingToHost)
// lo rechazaba y SyncGpuCleanBacking descargaba bytes viejos de la GPU ENCIMA de lo que el juego acababa de escribir.
// Ahora: (1) en bloques sin vigilar el evaluador lee siempre la memoria real del juego (nunca descarga); (2) al entrar
// un bloque (AllocTag) el hilo de la GPU suelta lo que la GPU tuviera alli (sin descargar) e invalida las texturas de
// ese rango, y otra vez en la primera epoca siguiente (lo que el juego escribe tras pedir el bloque).
inline std::mutex                                  g_enter_mutex;
inline std::vector<std::pair<uint64_t, uint64_t>> g_enter; // rangos que salen de la vigilancia (hilo de la GPU)
inline std::atomic<uint64_t> g_ro_desc_reads {0};   // lecturas de descriptores/tablas en bloques sin vigilar (memoria real)
inline std::atomic<uint64_t> g_ro_sync_skips {0};   // descargas de la GPU evitadas en bloques sin vigilar
inline std::atomic<uint64_t> g_ro_epoch_invals {0}; // invalidaciones de cache (GPU suelta + texturas) por epoca
inline void PushEnter(uint64_t lo, uint64_t size) {
	const std::lock_guard l(g_enter_mutex);
	g_enter.emplace_back(lo, size);
}
inline void TakeEnter(std::vector<std::pair<uint64_t, uint64_t>>& out) {
	const std::lock_guard l(g_enter_mutex);
	out.swap(g_enter);
	g_enter.clear();
}

struct RoList {
	uint64_t callers[16] {};
	uint32_t tags[16] {};
	int      nc = 0, nt = 0;
};
[[nodiscard]] inline const RoList& Ro() noexcept {
	static const RoList list = [] {
		RoList      l;
		const char* v = std::getenv("KYTY_FRAME_NATIVO2_RO");
		if (v == nullptr || v[0] == '\0') v = "0xaa2152,0xaa20cb";
		for (const char* p = v; *p != '\0';) {
			while (*p == ',' || *p == ' ' || *p == ';') ++p;
			if (*p == '\0') break;
			char* end = nullptr;
			if (*p == 't' || *p == 'T') {
				const unsigned long t = std::strtoul(p + 1, &end, 10);
				if (end != p + 1 && l.nt < 16) l.tags[l.nt++] = static_cast<uint32_t>(t);
			} else {
				const unsigned long long c = std::strtoull(p, &end, 16);
				if (end != p && l.nc < 16) l.callers[l.nc++] = c;
			}
			p = (end != nullptr && end != p) ? end : p + 1;
		}
		return l;
	}();
	return list;
}
[[nodiscard]] inline bool RoInstance(int i) noexcept {
	if (i < 0 || i >= ArenaDueno::kInstances || g_ro_bad[i].load(std::memory_order_relaxed)) return false;
	const RoList& l = Ro();
	for (int k = 0; k < l.nc; ++k)
		if (ArenaDueno::g_inst[i].caller == l.callers[k]) return true;
	for (int k = 0; k < l.nt; ++k)
		if (ArenaDueno::g_inst[i].tag == l.tags[k]) return true;
	return false;
}

[[nodiscard]] inline uint32_t RoHash(uint64_t g) noexcept { return static_cast<uint32_t>((g * 0x9E3779B97F4A7C15ull) >> 52); }
// Ranura del granulo g (la crea si add), o -1.
[[nodiscard]] inline int RoSlot(uint64_t g, bool add) noexcept {
	for (uint32_t s = RoHash(g), probe = 0; probe < kRoSlots; ++probe, s = (s + 1) & (kRoSlots - 1)) {
		uint64_t k = g_ro_key[s].load(std::memory_order_acquire);
		if (k == 0) {
			if (!add) return -1;
			if (g_ro_key[s].compare_exchange_strong(k, g, std::memory_order_acq_rel)) k = g;
		}
		if (k == g) return static_cast<int>(s);
	}
	return -1;
}
// Granulo sin vigilar (estado 1 o 2).
[[nodiscard]] inline bool RoGranuleUnwatched(uint64_t g) noexcept {
	const int s = RoSlot(g, false);
	return s >= 0 && g_ro_state[s].load(std::memory_order_acquire) != RoWatched;
}
// Algun granulo sin vigilar en [vaddr, vaddr + size) (rechazo rapido por rango; 0 granulos: falso al instante).
[[nodiscard]] inline bool RoAny(uint64_t vaddr, uint64_t size) noexcept {
	if (g_ro_granules.load(std::memory_order_relaxed) <= 0 || size == 0) return false;
	const uint64_t lo = (std::max)(vaddr, g_ro_lo.load(std::memory_order_relaxed));
	const uint64_t hi = (std::min)(vaddr + size, g_ro_hi.load(std::memory_order_relaxed));
	if (lo >= hi) return false;
	for (uint64_t g = lo >> kRoShift; g <= (hi - 1) >> kRoShift; ++g)
		if (RoGranuleUnwatched(g)) return true;
	return false;
}
// Todo [vaddr, vaddr + size) sin vigilar (la copia por epoca solo para esos rangos).
[[nodiscard]] inline bool RoAll(uint64_t vaddr, uint64_t size) noexcept {
	if (g_ro_granules.load(std::memory_order_relaxed) <= 0 || size == 0) return false;
	if (vaddr < g_ro_lo.load(std::memory_order_relaxed) || vaddr + size > g_ro_hi.load(std::memory_order_relaxed)) return false;
	for (uint64_t g = vaddr >> kRoShift; g <= (vaddr + size - 1) >> kRoShift; ++g)
		if (!RoGranuleUnwatched(g)) return false;
	return true;
}
// Primer tramo sin vigilar dentro de [at, end): [*b, *e). Falso si no hay (ProtectTransient).
[[nodiscard]] inline bool RoNextRun(uint64_t at, uint64_t end, uint64_t* b, uint64_t* e) noexcept {
	at  = (std::max)(at, g_ro_lo.load(std::memory_order_relaxed));
	end = (std::min)(end, g_ro_hi.load(std::memory_order_relaxed));
	if (at >= end) return false;
	const uint64_t gsize = uint64_t {1} << kRoShift;
	uint64_t       g     = at >> kRoShift;
	const uint64_t glast = (end - 1) >> kRoShift;
	while (g <= glast && !RoGranuleUnwatched(g)) ++g;
	if (g > glast) return false;
	uint64_t h = g;
	while (h + 1 <= glast && RoGranuleUnwatched(h + 1)) ++h;
	*b = (std::max)(at, g * gsize);
	*e = (std::min)(end, (h + 1) * gsize);
	return *b < *e;
}
inline void RoSet(uint64_t lo, uint64_t hi, uint8_t to) noexcept {
	for (uint64_t g = lo >> kRoShift; g < ((hi + (uint64_t {1} << kRoShift) - 1) >> kRoShift); ++g) {
		const int s = RoSlot(g, to != RoWatched);
		if (s < 0) continue;
		g_ro_gen[s].fetch_add(1, std::memory_order_acq_rel); // (v12: antes del estado nuevo)
		const uint8_t was = g_ro_state[s].exchange(to, std::memory_order_acq_rel);
		if (was == RoWatched && to != RoWatched) g_ro_granules.fetch_add(1, std::memory_order_relaxed);
		if (was != RoWatched && to == RoWatched) g_ro_granules.fetch_sub(1, std::memory_order_relaxed);
	}
	if (to == RoWatched) return;
	uint64_t v = g_ro_lo.load(std::memory_order_relaxed);
	while (lo < v && !g_ro_lo.compare_exchange_weak(v, lo, std::memory_order_relaxed)) {}
	v = g_ro_hi.load(std::memory_order_relaxed);
	while (hi > v && !g_ro_hi.compare_exchange_weak(v, hi, std::memory_order_relaxed)) {}
}
inline void PushRewatch(uint64_t lo, uint64_t size) {
	g_ro_rewatch.fetch_add(1, std::memory_order_relaxed);
	g_ro_rewatch_bytes.fetch_add(size, std::memory_order_relaxed);
	const std::lock_guard l(g_rewatch_mutex);
	g_rewatch.emplace_back(lo, size);
}
inline void TakeRewatch(std::vector<std::pair<uint64_t, uint64_t>>& out) {
	const std::lock_guard l(g_rewatch_mutex);
	out.swap(g_rewatch);
	g_rewatch.clear();
}

// Gancho AllocTag (tras ArenaDueno::OnAllocTag, que apunta el dueno). Devuelve true si [*lo, *lo + *size) debe quedar
// RW ya (instancia de solo lectura): el gancho llama a RestoreGuestWritable (tras marcar: ProtectTransient la descarta).
[[nodiscard]] inline bool OnAllocTagRo(uint64_t parent, uint64_t bytes, uint64_t result, uint64_t* lo, uint64_t* size) noexcept {
	if (result == 0 || parent == 0) return false;
	const uint64_t block = *reinterpret_cast<const volatile uint64_t*>(parent + 0x88);
	if (block == 0) return false;
	const uint64_t n = (bytes + block - 1) / block;
	if (n == 0 || n > 120) return false;
	*lo   = result;
	*size = n * block;
	// (v13, =6) Solo bloques dentro de un arena ya importado y con la cuarentena de FreeTag puesta (si no, vigilados).
	if (RoInstance(ArenaDueno::Owner(result)) && (!Nativo() || (Find(result, *size) >= 0 && NativoReady()))) {
		const bool already = RoAll(result, *size); // (ya RW: nada que desproteger)
		g_ro_allocs.fetch_add(1, std::memory_order_relaxed);
		g_ro_alloc_bytes.fetch_add(*size, std::memory_order_relaxed);
		if (!V12() && !Nativo()) g_epoch.fetch_add(1, std::memory_order_seq_cst); // (copias de este rango de antes: invalidas; v12: RoSet)
		RoSet(result, result + *size, RoUnwatched);
		PushEnter(result, *size); // (v11b) el hilo de la GPU suelta lo de la GPU e invalida texturas (por epoca)
		return !already;
	}
	// Instancia vigilada: si algun granulo estaba sin vigilar, vuelve a la vigilancia (CPU-modificado en el hilo GPU).
	if (RoAny(result, *size)) {
		RoSet(result, result + *size, RoWatched);
		PushRewatch(result, *size);
	}
	return false;
}
// Gancho FreeTag: los rangos exactos liberados que estaban sin vigilar quedan "libres sin vigilar" (siguen RW) hasta
// que AllocTag los de a alguien; la epoca sube (copias invalidas).
inline void OnFreedRo(uint64_t lo, uint64_t size) noexcept {
	if (!RoAny(lo, size)) return;
	g_ro_freed.fetch_add(size, std::memory_order_relaxed);
	for (uint64_t g = lo >> kRoShift; g < ((lo + size + (uint64_t {1} << kRoShift) - 1) >> kRoShift); ++g) {
		const int s = RoSlot(g, false);
		if (s < 0) continue;
		g_ro_gen[s].fetch_add(1, std::memory_order_acq_rel); // (v12: epoca del bloque)
		uint8_t want = RoUnwatched;
		g_ro_state[s].compare_exchange_strong(want, RoFreeUnwatched, std::memory_order_acq_rel);
	}
	if (!V12()) g_epoch.fetch_add(1, std::memory_order_seq_cst);
}
// Red de seguridad: escritura de la GPU en [vaddr, vaddr + size) sin vigilar. La instancia duena (o, sin dueno, los
// granulos tocados) vuelve a la vigilancia; los rangos devueltos se marcan CPU-modificados en la cache.
inline void OnUnexpectedGpuWrite(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>& out) {
	g_ro_unexpected.fetch_add(1, std::memory_order_relaxed);
	const int inst = ArenaDueno::Owner(vaddr);
	// (v13c, =6) Log 20261125-035522-917: UNA escritura legitima de la GPU (0x346e1c8e0, +87708, tag 30) devolvia la instancia
	// entera a la vigilancia para siempre (usos nativos 0/s desde entonces). Ahora solo vuelven los granulos que toca la
	// escritura (se detecta al traducir, antes de grabarla: desde ahi ese bloque lo sirve la cache con los bytes de la GPU);
	// cuando FreeTag lo suelte y AllocTag lo vuelva a dar, sale otra vez. La instancia entera solo tras 16 AVISOS.
	static std::atomic<uint32_t> s_hits[ArenaDueno::kInstances] {};
	const bool whole = !Nativo() || (inst >= 0 && s_hits[inst].fetch_add(1, std::memory_order_relaxed) + 1 >= 16);
	if (Nativo() && inst >= 0) {
		static std::atomic<uint32_t> s_printed {0};
		if (s_printed.fetch_add(1, std::memory_order_relaxed) < 32) {
			std::printf("[frame-nativo2 v13] AVISO por bloque: la GPU escribe 0x%llx (+%llu) en la instancia 0x%llx (llamador "
			            "eboot+0x%llx, tag %u): solo esos bloques vuelven a la vigilancia (aviso %u de esa instancia%s)\n",
			            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			            static_cast<unsigned long long>(ArenaDueno::g_inst[inst].self.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(ArenaDueno::g_inst[inst].caller), ArenaDueno::g_inst[inst].tag,
			            s_hits[inst].load(std::memory_order_relaxed), whole ? "; con 16 vuelve ENTERA" : "");
			std::fflush(stdout);
		}
	}
	if (whole && inst >= 0 && !g_ro_bad[inst].exchange(true, std::memory_order_relaxed)) {
		std::printf("[frame-nativo2 %s] AVISO: la GPU escribe 0x%llx (+%llu) en un bloque sin vigilar de la instancia 0x%llx "
		            "(llamador eboot+0x%llx, tag %u): esa instancia vuelve a la vigilancia normal\n",
		            V12() ? "v12" : "v11", static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
		            static_cast<unsigned long long>(ArenaDueno::g_inst[inst].self.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(ArenaDueno::g_inst[inst].caller), ArenaDueno::g_inst[inst].tag);
		std::fflush(stdout);
	}
	const uint64_t gsize = uint64_t {1} << kRoShift;
	for (int s = 0; s < kRoSlots; ++s) {
		const uint64_t g = g_ro_key[s].load(std::memory_order_acquire);
		if (g == 0 || g_ro_state[s].load(std::memory_order_acquire) == RoWatched) continue;
		const bool touched = g * gsize < vaddr + size && vaddr < (g + 1) * gsize;
		if (!touched && (!whole || inst < 0 || ArenaDueno::Owner(g * gsize) != inst)) continue;
		g_ro_gen[s].fetch_add(1, std::memory_order_acq_rel); // (v12)
		uint8_t was = g_ro_state[s].exchange(RoWatched, std::memory_order_acq_rel);
		if (was != RoWatched) {
			g_ro_granules.fetch_sub(1, std::memory_order_relaxed);
			out.emplace_back(g * gsize, gsize);
		}
	}
	g_epoch.fetch_add(1, std::memory_order_seq_cst);
}

// (FRAME-NATIVO2 v12) KYTY_FRAME_NATIVO2=5: SOLO LECTURA POR BLOQUES. Log 20261124-195714-534 (=4): la epoca GLOBAL subia
// en cada AllocTag/FreeTag de solo lectura (~840-880/s) y casi ninguna subida BDA se reutilizaba; las subidas BDA de hasta
// 1 MiB se salian a bloques vigilados (ventanas: [raiz] 121-126k VirtualProtect/s, 84k paginas sin acceso abiertas/s) y
// DrainEnter invalidaba texturas (1682-1765 rangos/s). Ghidra (eboot 01.007.000, VA del juego; archivo = +0xbdb0):
//  - AllocTag 0x7fe350 (archivo 0x80a100): n = ceil(bytes/[+0x88]) bloques; FUN_0x7fea60 solo da bloques con su bit
//    libre en la mascara de ocupados del trozo [c+0x18]; FreeTag 0x7fe540 (archivo 0x80a2f0) es el unico que la limpia
//    ([c+0x18] &= ~mascara de la clave). Un bloque es de UNA clave (tag<<32|idx) entre su AllocTag y su FreeTag.
//  - Alloc 0x811f60 (archivo 0x81dd10): cursor por hilo en [this+0xb8+t*0x30] (t < 0x14; si no [this+0x478] con
//    cerrojo) = {actual, restante}; devuelve (actual+align-1)&~(align-1) y deja actual = fin alineado a 8: SOLO CRECE.
//    El resto que no cabe (>0x18 bytes) va a una lista libre por hilo cuyos nodos se escriben DENTRO de ese resto (enlaces
//    y tamano: plVar1[0..2]) y se reparten despues desde ella. Los bytes ya entregados no se vuelven a entregar hasta
//    FreeTag; los NO entregados de un bloque (cola y restos) si se escriben despues.
// Por eso: (1) epoca POR BLOQUE (g_ro_gen) en vez de global; (2) toda copia/subida se recuerda tambien por el tick del
// planificador (como v7): una subida BDA es un rango CONSERVADOR (desplazamientos desconocidos: hasta 1 MiB) que incluye
// bytes aun no entregados, asi que no se puede reutilizar mas alla de la traduccion en curso; (3) las subidas BDA se
// recortan a los tramos sin vigilar (los bloques vigilados vecinos van por el camino normal); (4) sin invalidar
// texturas: al entrar un bloque solo se cuentan las imagenes que lo solapan y se invalidan solo las que no pinto la GPU.
// (v12, RESULTADO, log 20261124-210637-705) Sin parpadeos, escrituras GPU inesperadas 0, imagenes en los bloques 0,
// protecciones del juego 75-78k/s -> 40-45k/s y ventanas 0.05 ms/frame; pero 17-19 ms/frame (cpu 17 ms) frente a
// 12-13.5 ms sin FN2 (log 195230-139). Coste nuevo: 15.7-17 MB/frame de subidas BDA (~1 MiB por fallo: el plan BDA
// tiene desplazamientos desconocidos, BdaReadPlan.h 'Unknown offsets cover every U32 value', asi que el rango exacto
// no existe) y no se pueden reutilizar entre ticks sin detectar escrituras (Alloc 0x811f60 escribe despues en la parte
// no entregada del bloque). =5 queda APAGADO (solo con KYTY_FRAME_NATIVO2=5) y documentado.
// Suma de las epocas de los granulos de [vaddr, vaddr + size); falso si alguno esta vigilado (o el rango esta vacio).
[[nodiscard]] inline bool RoGen(uint64_t vaddr, uint64_t size, uint64_t* sum) noexcept {
	if (g_ro_granules.load(std::memory_order_relaxed) <= 0 || size == 0) return false;
	uint64_t total = 0;
	for (uint64_t g = vaddr >> kRoShift; g <= (vaddr + size - 1) >> kRoShift; ++g) {
		const int s = RoSlot(g, false);
		if (s < 0) return false;
		const uint32_t gen = g_ro_gen[s].load(std::memory_order_acquire);
		if (g_ro_state[s].load(std::memory_order_acquire) == RoWatched) return false;
		total += gen;
	}
	*sum = total;
	return true;
}
inline std::atomic<uint64_t> g_v12_copy_hits {0}, g_v12_copy_miss {0};      // copias V# por epoca de bloque
inline std::atomic<uint64_t> g_v12_bda_hits {0}, g_v12_bda_miss {0};        // subidas BDA (bloques sin vigilar)
inline std::atomic<uint64_t> g_v12_bda_bytes {0}, g_v12_bda_split {0};      // bytes subidos; rangos partidos
inline std::atomic<uint64_t> g_v12_enter_ranges {0}, g_v12_img_ranges {0};  // bloques que entran; con imagenes
inline std::atomic<uint64_t> g_v12_img_inval {0}, g_v12_img_gpu {0};        // invalidados (CPU); dejados (GPU)

// (FRAME-NATIVO2 v13) KYTY_FRAME_NATIVO2=6: NATIVO. El arena DefaultTagg se importa UNA vez en Vulkan (memoria del host
// coherente, como un PC) y los bloques de las instancias de solo lectura (tag 30 Init desde 0xaa2152, tag 28 desde
// 0xaa20cb; la GPU nunca escribe en ellos, medidor ARENA-DUENO) se leen EN VIVO desde ese VkBuffer: V# (ObtainBuffer) y
// BDA (entradas de la tabla de paginas apuntando al arena). Cero vigilancia, cero subidas, cero copias para ellos.
// Coherencia como en la consola: el juego no reescribe un byte entregado hasta FreeTag (Ghidra: Alloc 0x811f60 solo
// crece; FUN_0x7fea60, el buscador de AllocTag 0x7fe350, solo da bloques con su bit libre en [c+0x18..0x27]; FreeTag
// 0x7fe540 es el unico que los limpia: [c+0x18] &= ~[c+0x28+s*0x10], [padre+0x90] -= popcount, y si el trozo queda vacio
// lo devuelve con 0x801b80/0x801a10). En PS5 el juego espera su etiqueta de fin de GPU antes del NextFrame que llama a
// FreeTag; aqui las etiquetas se escriben AL TRADUCIR (sin C5: v4 con C5 dejaba el hilo de render esperando 26-34 ms),
// asi que esa espera no protege la lectura en vivo. Por eso FreeTag NO suelta estos bloques: se quitan de la mascara de
// la clave (siguen ocupados) y quedan en CUARENTENA hasta que (1) el hilo de la GPU haya traducido todo lo que el juego
// encolo antes del FreeTag (g_guest_consumed >= el de entonces + 2) y (2) la GPU de Vulkan haya completado el tick que
// grababa en ese momento (KnownGpuTick). Entonces se limpian sus bits bajo el cerrojo del juego (*[padre+0x80]) y
// [padre+0x90] baja. Es el "N fotogramas en vuelo" de un motor de PC: memoria de mas, ninguna espera.
// Resuelve v11 (estado de la GPU viejo en descriptores: al entrar un bloque se suelta lo de la GPU, como v11b/v12) y v12
// (subidas BDA de 1 MiB: ya no hay subida, la entrada BDA apunta al arena).
struct Held {
	uint64_t parent = 0, chunk = 0, base = 0, block = 0;
	uint64_t bits[2] {};
	uint64_t need = 0;  // g_guest_consumed que hay que alcanzar
	uint64_t tick = 0;  // tick de Vulkan grabandose cuando se alcanzo (0: aun no)
	uint64_t frame = 0; // g_guest_consumed al retener (estadistica)
};
inline std::mutex        g_held_mutex;
inline std::vector<Held> g_held;
inline std::atomic<uint64_t> g_n6_held_blocks {0}, g_n6_held_total {0}, g_n6_released {0}, g_n6_wait_frames {0};
inline std::atomic<uint64_t> g_n6_lost {0}, g_n6_max_held {0};
inline std::atomic<uint64_t> g_n6_native {0}, g_n6_calls {0}, g_n6_tsc {0}, g_n6_bda_native {0}, g_n6_bda_normal {0};
inline std::atomic<uint64_t> g_n6_enter_bda {0}, g_n6_restore_bda {0};
// (v13b) Fotogramas traducidos de margen antes de soltar un bloque retenido. Log 20261125-032303-388: con 2 se soltaban a
// los 2.0 fotogramas de media y habia un leve parpadeo; las colas de computo asincronas (ParQueues) se traducen en otro
// hilo y podrian ir un fotograma por detras del FrameBoundary del grafico. 3 deja ese fotograma de margen (~50 MB mas).
[[nodiscard]] inline uint64_t HoldFrames() noexcept {
	static const uint64_t n = [] {
		const char* v = std::getenv("KYTY_FRAME_NATIVO2_HOLD");
		const long  x = v != nullptr ? std::strtol(v, nullptr, 10) : 3;
		return static_cast<uint64_t>(x < 2 ? 2 : x > 8 ? 8 : x);
	}();
	return n;
}

// Llamado por CollectFreed con el cerrojo del padre tomado (hilo del juego, antes de FreeTag): los bloques de la clave
// que estan sin vigilar y dentro de un arena importado salen de la mascara de la clave y se retienen.
inline void HoldFreed(uint64_t parent, uint64_t chunk, uint64_t base, uint64_t block, int slot, uint64_t held0, uint64_t held1) noexcept {
	if (!NativoReady() || block == 0 || slot < 0) return;
	uint64_t m[2] = {held0, held1}, keep[2] = {0, 0};
	for (uint64_t i = 0; i < 120; ++i) {
		if (((m[i >> 6] >> (i & 63)) & 1u) == 0) continue;
		const uint64_t a = base + i * block;
		if (RoAll(a, block) && Find(a, block) >= 0) keep[i >> 6] |= uint64_t {1} << (i & 63);
	}
	if ((keep[0] | keep[1]) == 0) return;
	auto* lo = reinterpret_cast<volatile uint64_t*>(chunk + 0x28 + static_cast<uint64_t>(slot) * 0x10);
	auto* hi = reinterpret_cast<volatile uint64_t*>(chunk + 0x30 + static_cast<uint64_t>(slot) * 0x10);
	*lo      = *lo & ~keep[0];
	*hi      = *hi & ~keep[1];
	Held h;
	h.parent  = parent;
	h.chunk   = chunk;
	h.base    = base;
	h.block   = block;
	h.bits[0] = keep[0];
	h.bits[1] = keep[1];
	h.frame   = g_guest_consumed.load(std::memory_order_acquire);
	h.need    = h.frame + HoldFrames(); // (v13b: 3 por defecto; KYTY_FRAME_NATIVO2_HOLD=2..8)
	const uint64_t n = static_cast<uint64_t>(std::popcount(keep[0]) + std::popcount(keep[1]));
	const uint64_t now = g_n6_held_blocks.fetch_add(n, std::memory_order_relaxed) + n;
	g_n6_held_total.fetch_add(n, std::memory_order_relaxed);
	for (uint64_t mx = g_n6_max_held.load(std::memory_order_relaxed); now > mx && !g_n6_max_held.compare_exchange_weak(mx, now);) {}
	const std::lock_guard l(g_held_mutex);
	g_held.push_back(h);
}

// Hilo de la GPU: sella con el tick actual las retenciones cuyo trabajo ya esta traducido y devuelve las ya completadas.
// current(): tick de Vulkan que se graba ahora; completed(t): la GPU termino el tick t.
template <typename Current, typename Completed>
inline void TakeReleasable(std::vector<Held>& out, Current&& current, Completed&& completed) {
	const uint64_t consumed = g_guest_consumed.load(std::memory_order_acquire);
	const std::lock_guard l(g_held_mutex);
	if (g_held.empty()) return;
	size_t w = 0;
	for (size_t r = 0; r < g_held.size(); ++r) {
		Held& h = g_held[r];
		if (h.tick == 0 && consumed >= h.need) h.tick = current();
		if (h.tick != 0 && completed(h.tick)) {
			out.push_back(h);
			continue;
		}
		g_held[w++] = h;
	}
	g_held.resize(w);
}
// Suelta una retencion bajo el cerrojo del juego (el mismo protocolo 0 -> -1 que FreeTag). El trozo se busca otra vez en
// la lista del padre (si ya no esta, el padre se reinicio: nada que soltar). Despues los granulos pasan a "libres sin
// vigilar" (OnFreedRo), como si FreeTag los hubiera soltado ahora.
inline void Release(const Held& h) noexcept {
	const auto rd   = [](uint64_t a) { return *reinterpret_cast<const volatile uint64_t*>(a); };
	auto*      lock = reinterpret_cast<std::atomic<int64_t>*>(rd(h.parent + 0x80));
	if (lock != nullptr) {
		for (int64_t z = 0; !lock->compare_exchange_weak(z, -1, std::memory_order_acquire); z = 0) _mm_pause();
	}
	bool           found = false;
	const uint64_t heads = h.parent + 0x518;
	int            guard = 0;
	for (uint64_t hd = rd(heads); !found && hd != heads && hd != 0 && guard < 4096; hd = rd(hd), ++guard) {
		const uint64_t chunks = hd + 0x10;
		for (uint64_t c = rd(chunks); c != chunks && c != 0 && guard < 4096; c = rd(c), ++guard)
			if (c == h.chunk && rd(c + 0x10) == h.base) {
				found = true;
				break;
			}
	}
	const uint64_t n = static_cast<uint64_t>(std::popcount(h.bits[0]) + std::popcount(h.bits[1]));
	if (found) {
		auto* used0 = reinterpret_cast<volatile uint64_t*>(h.chunk + 0x18);
		auto* used1 = reinterpret_cast<volatile uint64_t*>(h.chunk + 0x20);
		*used0      = *used0 & ~h.bits[0];
		*used1      = *used1 & ~h.bits[1];
		auto* inuse = reinterpret_cast<volatile uint64_t*>(h.parent + 0x90);
		*inuse      = *inuse - n;
	}
	if (lock != nullptr) lock->store(0, std::memory_order_release);
	g_n6_held_blocks.fetch_sub(n, std::memory_order_relaxed);
	if (!found) {
		g_n6_lost.fetch_add(n, std::memory_order_relaxed);
		return;
	}
	g_n6_released.fetch_add(n, std::memory_order_relaxed);
	g_n6_wait_frames.fetch_add(n * (g_guest_consumed.load(std::memory_order_relaxed) - h.frame), std::memory_order_relaxed);
	for (uint64_t i = 0; i < 120; ++i)
		if (((h.bits[i >> 6] >> (i & 63)) & 1u) != 0) OnFreedRo(h.base + i * h.block, h.block);
}

// (v13) Coste de ObtainBuffer (rdtsc) y usos, solo con =6.
struct ObtainTimer {
	uint64_t start = 0;
	ObtainTimer() noexcept {
		if (Nativo()) start = __rdtsc();
	}
	~ObtainTimer() {
		if (start == 0) return;
		g_n6_calls.fetch_add(1, std::memory_order_relaxed);
		g_n6_tsc.fetch_add(__rdtsc() - start, std::memory_order_relaxed);
	}
	ObtainTimer(const ObtainTimer&)            = delete;
	ObtainTimer& operator=(const ObtainTimer&) = delete;
};

} // namespace FrameNativo2

