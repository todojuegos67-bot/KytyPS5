#pragma once
// (reentrante) Contadores por hilo para las mejoras que traducen DCBs (regs-rapido, bloques-rapido, indirect-tabla,
// trad1-prueba, recorrido-rapido, otros-rapido): prerrequisito de la fase 3 de par2 (varios carriles traduciendo a la
// vez). Cada hilo que cuenta tiene su bloque (en el montón, nunca liberado: sigue sumando aunque el hilo termine), al que
// llega por UN puntero thread_local; solo ese hilo escribe sus celdas (carga + almacenamiento relaxed, sin lock xadd),
// el hilo del informe las lee y suma todos los bloques. La ventana es la diferencia con la suma anterior (nadie pone a
// cero celdas de otro hilo). Con un hilo da exactamente los mismos numeros que el exchange(0) de antes.
// Coste en el camino caliente: lectura del puntero TLS (gs:[0x58] + tls_index en Windows, ~1 ns, L1) + prueba de nulo +
// suma; antes cada contador era un lock xadd (~5-8 ns sin contencion, mucho mas con dos hilos en la misma linea).
// Queda global a proposito en los 6 modulos: Mode() (static const, magic static seguro entre hilos), g_ab_on y el
// fotograma de indirect-tabla (interruptores atomicos: todos los carriles en la misma ventana A/B) y el historial de
// ventanas A/B (static w[4096], solo el hilo del informe).
// Quien cuenta mucho en un bucle cachea el bloque: `auto& c = Reent::Local<Tag, N>();` una vez por DCB/lista.
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace Reent {

template <class Tag, uint32_t N>
struct Block {
	std::atomic<uint64_t> cell[N] {};
	void Add(uint32_t i, uint64_t v = 1) {
		cell[i].store(cell[i].load(std::memory_order_relaxed) + v, std::memory_order_relaxed);
	}
};

template <class Tag, uint32_t N>
struct Registry {
	std::mutex                     lock;
	std::vector<Block<Tag, N>*>    blocks;
	uint64_t                       last[N] {}; // suma en la ventana anterior (solo el hilo del informe)
};

template <class Tag, uint32_t N>
[[nodiscard]] inline Registry<Tag, N>& Reg() {
	static Registry<Tag, N> r;
	return r;
}

template <class Tag, uint32_t N>
inline thread_local Block<Tag, N>* t_block = nullptr; // inicializacion constante: sin guarda por acceso

template <class Tag, uint32_t N>
inline Block<Tag, N>& Create() {
	auto* b = new Block<Tag, N>(); // vive hasta el final del proceso
	auto& r = Reg<Tag, N>();
	{
		std::lock_guard<std::mutex> g(r.lock);
		r.blocks.push_back(b);
	}
	t_block<Tag, N> = b;
	return *b;
}

template <class Tag, uint32_t N>
[[nodiscard]] inline Block<Tag, N>& Local() {
	auto* b = t_block<Tag, N>;
	return b != nullptr ? *b : Create<Tag, N>();
}

template <class Tag, uint32_t N>
inline void Add(uint32_t i, uint64_t v = 1) {
	Local<Tag, N>().Add(i, v);
}

// Hilo del informe: lo de esta ventana (suma de todos los carriles/hilos menos la suma anterior).
template <class Tag, uint32_t N>
inline void Take(uint64_t (&out)[N]) {
	auto&    r = Reg<Tag, N>();
	uint64_t sum[N] {};
	{
		std::lock_guard<std::mutex> g(r.lock);
		for (auto* b: r.blocks)
			for (uint32_t i = 0; i < N; ++i) sum[i] += b->cell[i].load(std::memory_order_relaxed);
	}
	for (uint32_t i = 0; i < N; ++i) out[i] = sum[i] - r.last[i], r.last[i] = sum[i];
}

// Indice de una celda con la sintaxis de antes (Modulo::Note(Modulo::g_taken)).
struct Slot {
	uint32_t i;
};

} // namespace Reent
