#pragma once
// (faltas3) KYTY_FALTAS_RECICLADO (sin variable o 1: encendido desde faltas3b; 0 apagado; 2 A/B dentro de la partida, cambia cada
// ventana de 5 s de KYTY_TECHO). Log T50 20261127-111950-777: la falta de escritura normal en DefaultTagg cuesta
// ~430 us en el hilo de envio, casi todo BufferCache::InvalidateMemory -> ReadMemory: lee de vuelta lo que la GPU
// escribio en la pagina para no perderlo, antes de que la CPU escriba (10.9 ms/fot en 25 faltas).
// Ghidra (eboot 01.007.000) FUN_00811f60, TaggedFrameAllocator::Allocate (this, bytes, align): asignador "bump" por
// hueco (this+0xb8+i*0x30, i = FUN_00832c80() < 0x14; si no, this+0x478 con cerrojo): [+0] cursor, [+8] lo que queda;
// devuelve el cursor alineado (uVar5) y lo avanza. Al acabarse el trozo, el resto (> 0x17 B) va a la lista libre del
// hueco (nodo escrito DENTRO de la memoria libre: 0x812046) y toma otro de la lista (LAB_0081209f) o uno nuevo
// (vtable +0xf8). Lo que devuelve es memoria que nadie usa desde ese momento: su contenido anterior esta muerto (el
// nuevo dueno lo escribe antes de que nada lo lea).
// La regla: cada reserva (gancho ya instalado de 0x811f60, OnFrameAllocAfter) marca muertos los granulos de 8 B que
// cubre ENTEROS; cada escritura de la GPU que el seguimiento anota (MemoryTracker::MarkRegionAsGpuModified y la subida
// con is_written) borra las marcas de los granulos que toca (aunque sea en parte). Una pagina con los 512 granulos
// marcados tiene TODO lo que la GPU escribio en ella reservado de nuevo despues: muerto. Solo entonces su falta de
// escritura no lee de vuelta: BufferCache::InvalidateMemoryForWrite de la pagina entera (descarta lo de la GPU, sin
// esperarla). Cualquier otra pagina (sin entrada, con un granulo sin marcar, sin escritura de GPU vista) sigue igual.
// Una carrera solo puede quitar marcas (mas lecturas de vuelta), nunca ponerlas sobre bytes vivos: la GPU solo escribe
// rangos vivos, y el juego no vuelve a reservar lo que la GPU aun escribe.
// Verificacion (siempre que esta encendido): 1 de cada 16 candidatas hace la lectura de vuelta normal y cuenta los
// bytes que trajo (lo que se habria descartado). Es informativo: esos bytes son de reservas muertas. Lo que si apaga
// todo para siempre: una candidata cuya pagina tiene bytes de la GPU que la tabla nunca vio escribir (g_miss).
#include <algorithm>
#include <atomic>
#include <new>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace FaltasReciclado {

// (sync2, KYTY_TECHO) Reparto del tramo "buffers" de la falta de escritura: la falta pone t_medir y
// MemoryTracker::InvalidateRegion suma la espera de los cerrojos de region y el on_flush (lectura de vuelta /
// descarte, con su SendCommandSync); el resto es trabajo propio. Ciclos.
inline thread_local bool     t_medir = false;
inline std::atomic<uint64_t> g_inv_lock {0}, g_inv_flush {0}, g_inv_regions {0};

inline std::atomic<int>  g_mode {-1};
inline std::atomic<bool> g_now {false}, g_off {false};
[[nodiscard]] inline int Mode() noexcept {
	int m = g_mode.load(std::memory_order_relaxed);
	if (m >= 0) return m;
	const char* v = std::getenv("KYTY_FALTAS_RECICLADO");
	// (faltas3b) Encendido por defecto: T52 (log 20261127-120905-287, Boletaria hasta el pabellon) A/B en la misma
	// partida 10.93 -> 9.40 ms/fot de faltas en el hilo de envio (-1.53), 13094 faltas sin lectura de vuelta, 0 fallos de
	// seguridad. =0 lo apaga; =2 A/B.
	m             = (v != nullptr && v[0] == '0') ? 0 : (v != nullptr && v[0] == '2') ? 2 : 1;
	g_now.store(m == 1, std::memory_order_relaxed);
	g_mode.store(m, std::memory_order_relaxed);
	std::printf("[faltas-reciclado] KYTY_FALTAS_RECICLADO=%d: %s\n", m,
	            m == 1 ? "encendido: falta de escritura en memoria de frame reservada de nuevo entera tras la GPU sin lectura de vuelta"
	                   : m == 2 ? "A/B cada 5 s (necesita KYTY_TECHO=1)" : "apagado (como antes)");
	std::fflush(stdout);
	return m;
}

constexpr size_t Slots = size_t {1} << 17u; // 128 Ki paginas (512 MiB de memoria de frame), 72 B cada una
struct Entry {
	std::atomic<uint64_t> key {0};      // pagina + 1
	std::atomic<uint64_t> gen {0};      // escrituras de GPU vistas
	std::atomic<uint64_t> bits[8] {};   // 512 granulos de 8 B: muerto
};
inline std::atomic<Entry*> g_table {nullptr};
inline std::atomic<uint64_t> g_full_lost {0};

[[nodiscard]] inline Entry* Table() noexcept {
	Entry* t = g_table.load(std::memory_order_acquire);
	if (t != nullptr || Mode() == 0) return t;
	static std::atomic<bool> once {false};
	if (once.exchange(true)) return g_table.load(std::memory_order_acquire);
	t = new (std::nothrow) Entry[Slots];
	g_table.store(t, std::memory_order_release);
	return t;
}
[[nodiscard]] inline Entry* Find(uint64_t page, bool create) noexcept {
	Entry* t = Table();
	if (t == nullptr) return nullptr;
	const uint64_t key = page + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 40u) % Slots;
	for (size_t p = 0; p < 32; ++p, i = (i + 1) % Slots) {
		uint64_t cur = t[i].key.load(std::memory_order_acquire);
		if (cur == key) return &t[i];
		if (cur == 0) {
			if (!create) return nullptr;
			if (t[i].key.compare_exchange_strong(cur, key, std::memory_order_acq_rel) || cur == key) return &t[i];
		}
	}
	if (create) g_full_lost.fetch_add(1, std::memory_order_relaxed);
	return nullptr;
}

// Granulos [lo, hi) de la pagina: set o clear.
inline void Apply(Entry* e, uint64_t lo, uint64_t hi, bool set) noexcept {
	while (lo < hi) {
		const uint64_t w = lo >> 6u, b = lo & 63u, n = std::min<uint64_t>(64 - b, hi - lo);
		const uint64_t mask = (n == 64 ? ~uint64_t {0} : ((uint64_t {1} << n) - 1u)) << b;
		if (set) e->bits[w].fetch_or(mask, std::memory_order_acq_rel);
		else e->bits[w].fetch_and(~mask, std::memory_order_acq_rel);
		lo += n;
	}
}

// OnFrameAllocAfter (hilo del juego): [address, address + bytes) acaba de entregarse.
inline void Reserved(uint64_t address, uint64_t bytes) noexcept {
	if (Mode() == 0 || g_off.load(std::memory_order_relaxed) || bytes == 0) return;
	const uint64_t lo_g = (address + 7u) >> 3u, hi_g = (address + bytes) >> 3u; // granulos enteros
	if (lo_g >= hi_g) return;
	for (uint64_t g = lo_g; g < hi_g;) {
		const uint64_t page = g >> 9u, end = std::min<uint64_t>(hi_g, (page + 1u) << 9u);
		if (Entry* e = Find(page, true); e != nullptr) Apply(e, g & 511u, ((end - 1u) & 511u) + 1u, true);
		g = end;
	}
}

// Escritura de la GPU anotada por el seguimiento (cualquier hilo).
inline void GpuWrite(uint64_t vaddr, uint64_t size) noexcept {
	if (g_table.load(std::memory_order_relaxed) == nullptr || size == 0) return;
	const uint64_t lo_g = vaddr >> 3u, hi_g = (vaddr + size + 7u) >> 3u; // granulos tocados
	for (uint64_t g = lo_g; g < hi_g;) {
		const uint64_t page = g >> 9u, end = std::min<uint64_t>(hi_g, (page + 1u) << 9u);
		if (Entry* e = Find(page, false); e != nullptr) {
			Apply(e, g & 511u, ((end - 1u) & 511u) + 1u, false);
			e->gen.fetch_add(1, std::memory_order_acq_rel);
		}
		g = end;
	}
}

// Contadores: candidatas, saltadas, verificadas, bytes que trajo la verificacion, fallos (apagan), no candidatas.
inline std::atomic<uint64_t> g_cand {0}, g_skip {0}, g_verify {0}, g_verify_bytes {0}, g_miss {0}, g_partial {0};

[[nodiscard]] inline bool Active() noexcept {
	return Mode() != 0 && g_now.load(std::memory_order_relaxed) && !g_off.load(std::memory_order_relaxed);
}
// La pagina de `vaddr`: todo lo que la GPU escribio en ella esta reservado de nuevo despues (y la GPU la escribio).
[[nodiscard]] inline bool PageDead(uint64_t vaddr) noexcept {
	Entry* e = Find(vaddr >> 12u, false);
	if (e == nullptr) return false;
	if (e->gen.load(std::memory_order_acquire) == 0) {
		g_partial.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	for (auto& w: e->bits)
		if (w.load(std::memory_order_acquire) != ~uint64_t {0}) {
			g_partial.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
	return true;
}
[[nodiscard]] inline bool VerifyThis() noexcept {
	return (g_cand.fetch_add(1, std::memory_order_relaxed) & 15u) == 0;
}
inline void Miss() noexcept {
	g_miss.fetch_add(1, std::memory_order_relaxed);
	if (!g_off.exchange(true)) {
		std::printf("[faltas-reciclado] SEGURIDAD: bytes de la GPU no vistos por la tabla en una candidata: APAGADO para el resto de la partida\n");
		std::fflush(stdout);
	}
}

// Desde TechoFaltas::Report.
inline void Report(char* out, size_t size) noexcept {
	static uint64_t l[6] {};
	const uint64_t  v[6] = {g_cand.load(), g_skip.load(), g_verify.load(), g_verify_bytes.load(), g_miss.load(), g_partial.load()};
	uint64_t        d[6];
	for (int i = 0; i < 6; ++i) d[i] = v[i] - l[i], l[i] = v[i];
	std::snprintf(out, size,
	              " | reciclado (modo %d%s): candidatas %llu, sin lectura %llu, verificadas %llu (bytes traidos %llu: de reservas muertas),"
	              " no candidatas %llu, SEGURIDAD fallos %llu, tabla llena %llu",
	              Mode(), g_off.load() ? ", APAGADO por seguridad" : "", static_cast<unsigned long long>(d[0]),
	              static_cast<unsigned long long>(d[1]), static_cast<unsigned long long>(d[2]), static_cast<unsigned long long>(d[3]),
	              static_cast<unsigned long long>(d[5]), static_cast<unsigned long long>(d[4]),
	              static_cast<unsigned long long>(g_full_lost.load()));
}

} // namespace FaltasReciclado
