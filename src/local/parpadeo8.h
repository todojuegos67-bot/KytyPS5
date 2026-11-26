#pragma once
// (parpadeo8) Lecturas PELIGROSAS de los bits del XPR en 0xa1f290 y su asentado con la GPU (w21).
//
// Ghidra (projelf_par1, w21):
//   - 0xa0d2d0 (llamada por 0xa0de50 en CADA frame de cada vista) pide un bloque NUEVO de 0x2daf0 bytes al heap
//     (0x80a510 -> 0x7f9cf0, arena 0x2f64408), lo pone a 0 y lo guarda en view+0x58. En ese bloque estan la tabla de
//     punteros a bits (fila obj*0x2d0, columna = tipo de pase) y los slots de pase en +0x2cd30 + idx*0x28.
//     => idx (0x28..0x2b en 0xa0a630 / 0xa05430) es el NUMERO DE PASE, no un indice de frame: el juego no tiene doble ni
//        triple buffer de bits por frame; cada frame tiene su bloque propio.
//   - Todos los slots apuntan a UNA etiqueta compartida por vista: *(view+0x120)+0x120 (y +0x1c0 para el otro grupo),
//     con view+0x120 = *(camara+0xe8)+0x18 (0xa11d70 / 0xa11940): objeto PERSISTENTE, el mismo en todos los frames.
//   - 0xa0de50 la pone a 0 (0xa0e2ad) y encola "XPRPreRender"; 0xab6080 espera *etiqueta == 1 (0x8351b0 -> 0x83a760
//     modo 0) y luego 0xa1f290 lee los bits por el puntero de la tabla.
//   - El emulador escribe esa etiqueta al TRADUCIR el RELEASE_MEM, antes de que la GPU corra el dispatch. Lo que el juego
//     lea despues es correcto solo si las paginas de los bits ya estan protegidas (escritas por GPU: la lectura falla y
//     se lee de vuelta) o ya leidas de vuelta en este frame. Si no: lee bytes que la GPU aun no ha escrito (los del
//     frame anterior en esa memoria del heap) y dibuja/oculta objetos que no tocan => parpadeo.
//
// Lo que hace:
//   1) Cuenta (siempre que parpadeo6 este activo, coste: unas lecturas atomicas por llamada) las lecturas de 0xa1f290
//      cuyas paginas de bits NO estan protegidas, NO se han leido de vuelta desde que este frame creo la tabla de la
//      vista, y el asentado no encontro nada de la GPU => "PELIGROSAS". Se separan en: con un cull ya traducido que la
//      GPU aun no ha corrido (la GPU no ha escrito todavia) y sin cull pendiente (bits nunca vistos como de GPU).
//   2) KYTY_PARPADEO8=1: en una lectura peligrosa con cull pendiente, el hilo del juego espera (max 4 ms) a que la GPU
//      corra el ultimo cull traducido y vuelve a asentar el rango (asienta el slot correcto). Sin parches del juego.
// Linea "[parpadeo8]" cada 5 s (la saca el hilo del juego, con la de [parpadeo6]).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace Parpadeo8 {

// KYTY_PARPADEO8=1: la espera en lecturas peligrosas. El contador va siempre (con parpadeo6).
[[nodiscard]] inline bool FixOn() noexcept {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PARPADEO8");
		return v != nullptr && v[0] == '1' && v[1] == 0;
	}();
	return on;
}

inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- Lado GPU (graphicsRun.cpp): el ultimo cull traducido y si la GPU ya lo corrio ----
// Tick de la GPU que contendra el ultimo cull traducido (CurrentTick() al escribir su etiqueta: el buffer abierto se
// cierra con ese tick o uno mayor). 0 = ninguno.
inline std::atomic<uint64_t> g_cull_tick {0};
// IsFree del MasterSemaphore, puesto por graphicsRun.cpp (asi este fichero no necesita los de Vulkan).
using IsFreeFn = bool (*)(const void* master, uint64_t tick);
inline std::atomic<IsFreeFn>    g_is_free {nullptr};
inline std::atomic<const void*> g_master {nullptr};

// Hilo GPU, al escribir la etiqueta de un cull (serie1: Classify == Cull).
inline void NoteCullLabel(const void* master, IsFreeFn is_free, uint64_t current_tick) noexcept {
	g_master.store(master, std::memory_order_relaxed);
	g_is_free.store(is_free, std::memory_order_relaxed);
	g_cull_tick.store(current_tick, std::memory_order_release);
}
// Hilo del juego: tick del cull pendiente en la GPU, o 0 si la GPU ya lo corrio (o no hay).
[[nodiscard]] inline uint64_t PendingCullTick() noexcept {
	const uint64_t tick = g_cull_tick.load(std::memory_order_acquire);
	const IsFreeFn fn   = g_is_free.load(std::memory_order_relaxed);
	const void*    m    = g_master.load(std::memory_order_relaxed);
	if (tick == 0 || fn == nullptr || m == nullptr) return 0;
	return fn(m, tick) ? 0 : tick;
}

// ---- Lado juego: que paginas de bits se han leido de vuelta y desde cuando existe la tabla de la vista ----
inline constexpr uint32_t PageSlots = 1u << 16; // directo, con etiqueta (perdidas = cuenta de mas, nunca de menos)
inline std::atomic<uint64_t> g_page_tag[PageSlots];
inline std::atomic<int64_t>  g_page_ns[PageSlots];
inline constexpr uint32_t TableSlots = 256;
inline std::atomic<uint64_t> g_table[TableSlots];
inline std::atomic<int64_t>  g_table_ns[TableSlots];

inline uint32_t Mix(uint64_t v) noexcept {
	v ^= v >> 33;
	v *= 0xff51afd7ed558ccdull;
	v ^= v >> 33;
	return static_cast<uint32_t>(v);
}
// Primera vez que se ve el bloque de la vista (view+0x58, nuevo cada frame): su hora.
inline int64_t TableSince(uint64_t table, int64_t now) noexcept {
	const uint32_t i   = Mix(table) & (TableSlots - 1u);
	uint64_t       cur = g_table[i].load(std::memory_order_acquire);
	if (cur == table) return g_table_ns[i].load(std::memory_order_acquire);
	// Otro bloque en el slot (o vacio): este es nuevo. Carrera entre hilos: gana uno; el otro lee su hora.
	if (g_table[i].compare_exchange_strong(cur, table, std::memory_order_acq_rel)) {
		g_table_ns[i].store(now, std::memory_order_release);
		return now;
	}
	return cur == table ? g_table_ns[i].load(std::memory_order_acquire) : now;
}
inline void StampPages(uint64_t at, uint64_t bytes, int64_t now) noexcept {
	for (uint64_t p = at >> 12, e = (at + bytes + 4095u) >> 12; p < e; ++p) {
		const uint32_t i = Mix(p) & (PageSlots - 1u);
		g_page_ns[i].store(now, std::memory_order_relaxed);
		g_page_tag[i].store(p, std::memory_order_release);
	}
}
// Todas las paginas leidas de vuelta despues de `since`.
[[nodiscard]] inline bool PagesSince(uint64_t at, uint64_t bytes, int64_t since) noexcept {
	for (uint64_t p = at >> 12, e = (at + bytes + 4095u) >> 12; p < e; ++p) {
		const uint32_t i = Mix(p) & (PageSlots - 1u);
		if (g_page_tag[i].load(std::memory_order_acquire) != p || g_page_ns[i].load(std::memory_order_relaxed) < since) return false;
	}
	return true;
}

struct Counters {
	std::atomic<uint64_t> calls {0};          // lecturas miradas
	std::atomic<uint64_t> gpu_owned {0};      // el asentado encontro bytes de GPU (protegidas / guardadas): seguras
	std::atomic<uint64_t> already {0};        // paginas ya leidas de vuelta en este frame: seguras
	std::atomic<uint64_t> danger {0};         // PELIGROSAS: ni de GPU ni leidas de vuelta en este frame
	std::atomic<uint64_t> danger_pending {0}; // ... con un cull traducido que la GPU aun no ha corrido
	std::atomic<uint64_t> fixed {0};          // (KYTY_PARPADEO8=1) esperado al cull y asentado de nuevo
	std::atomic<uint64_t> fixed_gpu {0};      // ... y entonces si habia bytes de GPU (el arreglo cambio lo leido)
	std::atomic<uint64_t> fix_left {0};       // ... la GPU no lo corrio en 4 ms
	std::atomic<uint64_t> fix_us {0};
};
inline Counters              g_c;
inline std::atomic<int64_t>  g_last_ns {0};

// Tras el asentado de parpadeo6 (hilo del juego). `gpu_flags`: los RaizSettle* de Busy|Pending|Readback que vio.
// `resettle`: vuelve a asentar el rango y devuelve sus flags (para el arreglo). Devuelve true si fue peligrosa.
template <typename Resettle>
inline bool Check(uint64_t table, uint64_t bits_at, uint64_t bytes, bool gpu_seen, Resettle&& resettle) noexcept {
	auto&         c     = g_c;
	const int64_t now   = NowNs();
	const int64_t since = TableSince(table, now);
	c.calls.fetch_add(1, std::memory_order_relaxed);
	if (gpu_seen) {
		c.gpu_owned.fetch_add(1, std::memory_order_relaxed);
		StampPages(bits_at, bytes, now);
		return false;
	}
	if (PagesSince(bits_at, bytes, since)) {
		c.already.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	c.danger.fetch_add(1, std::memory_order_relaxed);
	const uint64_t tick = PendingCullTick();
	if (tick != 0) {
		c.danger_pending.fetch_add(1, std::memory_order_relaxed);
		if (FixOn()) {
			const int64_t until = now + 4'000'000;
			while (PendingCullTick() == tick && NowNs() < until) std::this_thread::yield();
			const bool left = PendingCullTick() == tick;
			const bool gpu  = resettle();
			c.fixed.fetch_add(1, std::memory_order_relaxed);
			if (gpu) c.fixed_gpu.fetch_add(1, std::memory_order_relaxed);
			if (left) c.fix_left.fetch_add(1, std::memory_order_relaxed);
			c.fix_us.fetch_add(static_cast<uint64_t>((NowNs() - now) / 1000), std::memory_order_relaxed);
			if (gpu) StampPages(bits_at, bytes, NowNs());
		}
	}
	return true;
}

inline void MaybeReport() noexcept {
	const int64_t now  = NowNs();
	int64_t       last = g_last_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_last_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 5'000'000'000ll || !g_last_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
	auto&      c    = g_c;
	const auto take = [](std::atomic<uint64_t>& a) { return static_cast<unsigned long long>(a.exchange(0, std::memory_order_relaxed)); };
	const auto calls = take(c.calls), gpu = take(c.gpu_owned), already = take(c.already), danger = take(c.danger),
	           pend = take(c.danger_pending), fixed = take(c.fixed), fixed_gpu = take(c.fixed_gpu), left = take(c.fix_left),
	           fix_us = take(c.fix_us);
	std::printf("[parpadeo8] %.1f s: lecturas de bits en 0xa1f290 %llu | seguras: con bytes de GPU %llu, ya leidas de vuelta "
	            "este frame %llu | PELIGROSAS (pagina ni de GPU ni leida de vuelta en este frame) %llu, de ellas con cull "
	            "traducido sin correr en GPU %llu | arreglo (KYTY_PARPADEO8=%d): esperas %llu (%.2f ms), con bytes de GPU "
	            "despues %llu, sin correr en 4 ms %llu\n",
	            static_cast<double>(now - last) / 1e9, calls, gpu, already, danger, pend, FixOn() ? 1 : 0, fixed,
	            static_cast<double>(fix_us) / 1000.0, fixed_gpu, left);
	std::fflush(stdout);
}

} // namespace Parpadeo8
