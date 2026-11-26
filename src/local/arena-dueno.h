#pragma once
// (ARENA-DUENO) KYTY_ARENA_DUENO=1 (apagado por defecto): medidor. Que instancia de TaggedFrameAllocator es duena de
// cada bloque de los arenas DefaultTagg, y cuanto escribe / lee la GPU y cuantos [wb-guard] caen en los bloques de cada
// una. Es el dato que necesita la via A (mover a un arena aparte la instancia que escribe la GPU).
// Ghidra (eboot 01.007.000, archivo = juego + 0xbdb0):
//  - Init 0x8114a0 (archivo 0x81d250)(this, padre, num_buffers): [+0x88]=padre, [+0x94]=num_buffers. Gancho: this,
//    padre, num_buffers, tag [+0x30] y la direccion de retorno (el llamador).
//  - AllocTag del padre = vtable+0xf8 = juego 0x7fe350 (archivo 0x80a100; reloc 0x25acee0, la ranura antes de FreeTag
//    0x7fe540 en 0x25acee8)(padre, instancia, bytes, bloque, clave tag<<32|idx, origen): devuelve la base de
//    ceil(bytes/[padre+0x88]) bloques contiguos de 1 MiB. Lo llama Alloc 0x811f60 (archivo 0x81dd10) cuando la
//    instancia necesita bloque nuevo. Gancho: al volver, esos bloques pasan a esa instancia (tabla fija).
// La tabla va por granulos de 512 KiB (las bases de los arenas estan alineadas a 512 KiB, los bloques a la base + i MiB).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ArenaDueno {

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_ARENA_DUENO");
		return v != nullptr && (v[0] == '1' || v[0] == '2');
	}();
	return on;
}
// (ARENA-DUENO paso 0) KYTY_ARENA_DUENO=2: ademas el otro padre (0x221770ce0 en el log 20261125-032303-388, 20 instancias):
// AllocTag 0x7fcb50 / FreeTag 0x7fcd50 (vtable 0x25accc8 +0xf8/+0x100). Formato (Ghidra): mascaras de ranura [p+0x1ea98],
// claves [p+0x1e898+s*8], montones [p+0x1e888]; trozo c: base [c+0x10], ocupados [c+0x18] 0xf0 qwords (0x3c00 bloques),
// mascara de la clave c+0x798+s*0x780. Tamano de bloque [p+0x88] SIN MEDIR: este medidor lo dice. Solo mide.
[[nodiscard]] inline bool Dos() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_ARENA_DUENO");
		return v != nullptr && v[0] == '2';
	}();
	return on;
}

struct Instance {
	std::atomic<uint64_t> self {0};
	uint64_t              caller = 0, parent = 0; // (caller: desplazamiento en el eboot de la direccion de retorno de Init)
	uint32_t              nbuf = 0, tag = 0;
	std::atomic<uint64_t> allocs {0}, alloc_bytes {0};
	std::atomic<uint64_t> w_n {0}, w_b {0}, r_n {0}, r_b {0}, wbg {0};
	uint64_t              last_w_n = 0, last_w_b = 0, last_r_n = 0, last_r_b = 0, last_wbg = 0; // (solo el informe)
};
constexpr int                kInstances = 64;
inline Instance              g_inst[kInstances];
inline std::atomic<int>      g_inst_count {0};
inline std::atomic<bool>     g_hooked {false};
inline std::atomic<uint64_t> g_eboot {0};
inline std::atomic<uint64_t> g_lo {UINT64_MAX}, g_hi {0}; // rango cubierto por la tabla (rechazo rapido)
inline std::atomic<uint64_t> g_wbg_none {0}, g_init_calls {0}, g_alloc_calls {0};

// Granulo (512 KiB) -> indice de instancia + 1. Direccionamiento abierto, sin borrar (el dueno se sobrescribe).
constexpr int                kSlots = 4096;
constexpr int                kShift = 19;
inline std::atomic<uint64_t> g_key[kSlots] {};
inline std::atomic<uint32_t> g_owner[kSlots] {};

[[nodiscard]] inline uint32_t Hash(uint64_t g) noexcept { return static_cast<uint32_t>((g * 0x9E3779B97F4A7C15ull) >> 52); }

// Indice de la instancia (la crea si no estaba; -1 si la tabla esta llena).
inline int FindOrAdd(uint64_t self) noexcept {
	const int n = g_inst_count.load(std::memory_order_acquire);
	for (int i = 0; i < n; ++i)
		if (g_inst[i].self.load(std::memory_order_relaxed) == self) return i;
	static std::atomic<int> claim {0};
	for (;;) { // (pocas llamadas: Init 15 veces, AllocTag de vez en cuando)
		int expected = 0;
		if (claim.compare_exchange_weak(expected, 1, std::memory_order_acquire)) break;
	}
	int       idx = -1;
	const int m   = g_inst_count.load(std::memory_order_relaxed);
	for (int i = 0; i < m; ++i)
		if (g_inst[i].self.load(std::memory_order_relaxed) == self) idx = i;
	if (idx < 0 && m < kInstances) {
		idx = m;
		g_inst[idx].self.store(self, std::memory_order_relaxed);
		g_inst_count.store(m + 1, std::memory_order_release);
	}
	claim.store(0, std::memory_order_release);
	return idx;
}

inline void OnInit(uint64_t self, uint64_t parent, uint32_t nbuf, uint64_t ret) noexcept {
	g_init_calls.fetch_add(1, std::memory_order_relaxed);
	const int i = FindOrAdd(self);
	if (i < 0) return;
	const uint64_t eboot = g_eboot.load(std::memory_order_relaxed);
	g_inst[i].caller     = ret >= eboot ? ret - eboot : ret;
	g_inst[i].parent     = parent;
	g_inst[i].nbuf       = nbuf;
	g_inst[i].tag        = *reinterpret_cast<const volatile uint32_t*>(self + 0x30);
	std::printf("[arena-dueno] Init 0x%llx: padre 0x%llx, %u buferes, tag %u, llamador eboot+0x%llx (instancia %d)\n",
	            static_cast<unsigned long long>(self), static_cast<unsigned long long>(parent), nbuf, g_inst[i].tag,
	            static_cast<unsigned long long>(g_inst[i].caller), i);
	std::fflush(stdout);
}

inline void OnAllocTag(uint64_t parent, uint64_t self, uint64_t bytes, uint64_t key, uint64_t result) noexcept {
	g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
	if (result == 0 || parent == 0) return;
	const uint64_t block = *reinterpret_cast<const volatile uint64_t*>(parent + 0x88);
	if (block == 0) return;
	const uint64_t n = (bytes + block - 1) / block;
	if (n == 0 || n > 120) return;
	const int i = FindOrAdd(self);
	if (i < 0) return;
	if (g_inst[i].tag == 0) g_inst[i].tag = static_cast<uint32_t>(key >> 32u);
	if (g_inst[i].parent == 0) g_inst[i].parent = parent;
	g_inst[i].allocs.fetch_add(1, std::memory_order_relaxed);
	g_inst[i].alloc_bytes.fetch_add(n * block, std::memory_order_relaxed);
	const uint64_t lo = result, hi = result + n * block;
	for (uint64_t g = lo >> kShift; g < ((hi + (uint64_t {1} << kShift) - 1) >> kShift); ++g) {
		for (uint32_t s = Hash(g), probe = 0; probe < kSlots; ++probe, s = (s + 1) & (kSlots - 1)) {
			uint64_t k = g_key[s].load(std::memory_order_acquire);
			if (k == 0 && g_key[s].compare_exchange_strong(k, g, std::memory_order_acq_rel)) k = g;
			if (k == g) {
				g_owner[s].store(static_cast<uint32_t>(i + 1), std::memory_order_release);
				break;
			}
		}
	}
	uint64_t v = g_lo.load(std::memory_order_relaxed);
	while (lo < v && !g_lo.compare_exchange_weak(v, lo, std::memory_order_relaxed)) {}
	v = g_hi.load(std::memory_order_relaxed);
	while (hi > v && !g_hi.compare_exchange_weak(v, hi, std::memory_order_relaxed)) {}
}

// Instancia duena de la direccion, o -1.
[[nodiscard]] inline int Owner(uint64_t vaddr) noexcept {
	if (vaddr < g_lo.load(std::memory_order_relaxed) || vaddr >= g_hi.load(std::memory_order_relaxed)) return -1;
	const uint64_t g = vaddr >> kShift;
	for (uint32_t s = Hash(g), probe = 0; probe < kSlots; ++probe, s = (s + 1) & (kSlots - 1)) {
		const uint64_t k = g_key[s].load(std::memory_order_acquire);
		if (k == 0) return -1;
		if (k == g) return static_cast<int>(g_owner[s].load(std::memory_order_acquire)) - 1;
	}
	return -1;
}

// (paso 0) Tabla del otro padre, por granulos del tamano de SU bloque (potencia de 2, fijado en el primer AllocTag).
constexpr int                kSlotsB = 1 << 17;
inline std::atomic<uint64_t> g_keyB[kSlotsB] {};
inline std::atomic<uint32_t> g_ownerB[kSlotsB] {};
inline std::atomic<int>      g_shiftB {0};
inline std::atomic<uint64_t> g_blockB {0}, g_allocB {0}, g_allocB_bytes {0}, g_allocB_odd {0}, g_fullB {0};
inline std::atomic<uint64_t> g_parentB {0}, g_loB {UINT64_MAX}, g_hiB {0};
[[nodiscard]] inline uint32_t HashB(uint64_t g) noexcept { return static_cast<uint32_t>((g * 0x9E3779B97F4A7C15ull) >> 47); }
inline void OnAllocTagB(uint64_t parent, uint64_t self, uint64_t bytes, uint64_t key, uint64_t result) noexcept {
	g_allocB.fetch_add(1, std::memory_order_relaxed);
	if (result == 0 || parent == 0) return;
	const uint64_t block = *reinterpret_cast<const volatile uint64_t*>(parent + 0x88);
	if (block == 0 || (block & (block - 1)) != 0) {
		g_allocB_odd.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	int shift = 0;
	while ((uint64_t {1} << shift) < block) ++shift;
	int want = 0;
	if (!g_shiftB.compare_exchange_strong(want, shift, std::memory_order_acq_rel) && want != shift) {
		g_allocB_odd.fetch_add(1, std::memory_order_relaxed); // (otro tamano de bloque: no se apunta)
		return;
	}
	if (g_blockB.exchange(block, std::memory_order_relaxed) == 0) {
		g_parentB.store(parent, std::memory_order_relaxed);
		std::printf("[arena-dueno paso0] AllocTag 0x7fcb50: padre 0x%llx, bloque %llu bytes ([padre+0x88])\n",
		            static_cast<unsigned long long>(parent), static_cast<unsigned long long>(block));
		std::fflush(stdout);
	}
	const uint64_t n = (bytes + block - 1) / block;
	if (n == 0 || n > 0x3c00) return;
	const int i = FindOrAdd(self);
	if (i < 0) return;
	if (g_inst[i].tag == 0) g_inst[i].tag = static_cast<uint32_t>(key >> 32u);
	if (g_inst[i].parent == 0) g_inst[i].parent = parent;
	g_inst[i].allocs.fetch_add(1, std::memory_order_relaxed);
	g_inst[i].alloc_bytes.fetch_add(n * block, std::memory_order_relaxed);
	g_allocB_bytes.fetch_add(n * block, std::memory_order_relaxed);
	for (uint64_t g = result >> shift; g < ((result + n * block) >> shift); ++g) {
		bool put = false;
		for (uint32_t s = HashB(g), probe = 0; probe < 4096; ++probe, s = (s + 1) & (kSlotsB - 1)) {
			uint64_t k = g_keyB[s].load(std::memory_order_acquire);
			if (k == 0 && g_keyB[s].compare_exchange_strong(k, g, std::memory_order_acq_rel)) k = g;
			if (k == g) {
				g_ownerB[s].store(static_cast<uint32_t>(i + 1), std::memory_order_release);
				put = true;
				break;
			}
		}
		if (!put) g_fullB.fetch_add(1, std::memory_order_relaxed);
	}
	uint64_t v = g_loB.load(std::memory_order_relaxed);
	while (result < v && !g_loB.compare_exchange_weak(v, result, std::memory_order_relaxed)) {}
	v = g_hiB.load(std::memory_order_relaxed);
	while (result + n * block > v && !g_hiB.compare_exchange_weak(v, result + n * block, std::memory_order_relaxed)) {}
}
[[nodiscard]] inline int OwnerB(uint64_t vaddr) noexcept {
	const int shift = g_shiftB.load(std::memory_order_relaxed);
	if (shift == 0 || vaddr < g_loB.load(std::memory_order_relaxed) || vaddr >= g_hiB.load(std::memory_order_relaxed)) return -1;
	const uint64_t g = vaddr >> shift;
	for (uint32_t s = HashB(g), probe = 0; probe < 4096; ++probe, s = (s + 1) & (kSlotsB - 1)) {
		const uint64_t k = g_keyB[s].load(std::memory_order_acquire);
		if (k == 0) return -1;
		if (k == g) return static_cast<int>(g_ownerB[s].load(std::memory_order_acquire)) - 1;
	}
	return -1;
}

inline void Report() noexcept {
	using Clock = std::chrono::steady_clock;
	static std::atomic<int64_t> last {0};
	const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
	int64_t       was = last.load(std::memory_order_relaxed);
	if (was == 0) {
		last.compare_exchange_strong(was, now, std::memory_order_relaxed);
		return;
	}
	if (now - was < 10000 || !last.compare_exchange_strong(was, now, std::memory_order_relaxed)) return;
	const double secs = static_cast<double>(now - was) / 1000.0;
	const int    n    = g_inst_count.load(std::memory_order_acquire);
	uint32_t     granules[kInstances] {};
	for (int s = 0; s < kSlots; ++s)
		if (const uint32_t o = g_owner[s].load(std::memory_order_relaxed); o != 0 && o <= kInstances) ++granules[o - 1];
	std::printf("[arena-dueno] cada 10 s (KYTY_ARENA_DUENO=1): %d instancias, Init %llu, AllocTag %llu; wb-guard sin dueno %llu\n",
	            n, static_cast<unsigned long long>(g_init_calls.load()), static_cast<unsigned long long>(g_alloc_calls.load()),
	            static_cast<unsigned long long>(g_wbg_none.load()));
	uint32_t granulesB[kInstances] {};
	if (Dos()) {
		for (int s = 0; s < kSlotsB; ++s)
			if (const uint32_t o = g_ownerB[s].load(std::memory_order_relaxed); o != 0 && o <= kInstances) ++granulesB[o - 1];
		std::printf("[arena-dueno paso0] padre 0x%llx (AllocTag 0x7fcb50): bloque %llu bytes, AllocTag %llu (%.1f MB en total), "
		            "tamanos raros %llu, tabla llena %llu\n",
		            static_cast<unsigned long long>(g_parentB.load()), static_cast<unsigned long long>(g_blockB.load()),
		            static_cast<unsigned long long>(g_allocB.load()), static_cast<double>(g_allocB_bytes.load()) / 1e6,
		            static_cast<unsigned long long>(g_allocB_odd.load()), static_cast<unsigned long long>(g_fullB.load()));
	}
	for (int i = 0; i < n; ++i) {
		Instance&      x  = g_inst[i];
		const uint64_t wn = x.w_n.load(std::memory_order_relaxed), wb = x.w_b.load(std::memory_order_relaxed);
		const uint64_t rn = x.r_n.load(std::memory_order_relaxed), rb = x.r_b.load(std::memory_order_relaxed);
		const uint64_t gw = x.wbg.load(std::memory_order_relaxed);
		std::printf("[arena-dueno] instancia 0x%llx (llamador Init eboot+0x%llx, tag %u, buferes %u, padre 0x%llx, bloques %.1f MiB, del otro padre %.2f MiB): "
		            "GPU escribe %.0f veces/s (%.2f MB/s); lecturas GPU %.0f veces/s (%.2f MB/s); wb-guard %llu (total %llu)\n",
		            static_cast<unsigned long long>(x.self.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(x.caller), x.tag, x.nbuf, static_cast<unsigned long long>(x.parent),
		            granules[i] * 0.5,
		            static_cast<double>(granulesB[i]) * static_cast<double>(g_blockB.load(std::memory_order_relaxed)) / 1048576.0,
		            static_cast<double>(wn - x.last_w_n) / secs,
		            static_cast<double>(wb - x.last_w_b) / secs / 1e6, static_cast<double>(rn - x.last_r_n) / secs,
		            static_cast<double>(rb - x.last_r_b) / secs / 1e6, static_cast<unsigned long long>(gw - x.last_wbg),
		            static_cast<unsigned long long>(gw));
		x.last_w_n = wn;
		x.last_w_b = wb;
		x.last_r_n = rn;
		x.last_r_b = rb;
		x.last_wbg = gw;
	}
	std::fflush(stdout);
}

// Cache de buferes (ObtainBuffer): cada uso por la GPU.
inline void OnGpuUse(uint64_t vaddr, uint64_t size, bool written) noexcept {
	static std::atomic<uint32_t> tick {0};
	if ((tick.fetch_add(1, std::memory_order_relaxed) & 1023u) == 0) Report();
	int i = Owner(vaddr);
	if (i < 0 && Dos()) i = OwnerB(vaddr); // (paso 0: el otro padre)
	if (i < 0) return;
	if (written) {
		g_inst[i].w_n.fetch_add(1, std::memory_order_relaxed);
		g_inst[i].w_b.fetch_add(size, std::memory_order_relaxed);
	} else {
		g_inst[i].r_n.fetch_add(1, std::memory_order_relaxed);
		g_inst[i].r_b.fetch_add(size, std::memory_order_relaxed);
	}
}

// Cada evento [wb-guard] (pagina escrita por el juego mientras era de la GPU).
inline void OnWbGuard(uint64_t page) noexcept {
	const int i = Owner(page);
	if (i < 0)
		g_wbg_none.fetch_add(1, std::memory_order_relaxed);
	else
		g_inst[i].wbg.fetch_add(1, std::memory_order_relaxed);
	Report();
}

} // namespace ArenaDueno
