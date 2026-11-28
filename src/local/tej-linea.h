#pragma once
// (tej-linea) KYTY_TEJ: escrituras de los trabajadores de la sombra en lineas de cache que el hilo de render lee o
// escribe en cada draw (ping-pong entre CCD por Infinity Fabric, ~100-200 ns por fallo; banco
// src/local/bench/tej-linea-bench.cpp). Dato: build 0e327f1, escena pesada, B +18 Mciclos/fot de CPU del render con la
// misma frecuencia (4.22-4.24 GHz) y el coste cronometrado de la sombra en el render 0.16-0.18 ms: el resto es el mismo
// codigo del render mas lento (fallos de cache por coherencia).
// Lo que hay aqui (sin interruptor: no cambia nada de lo que se dibuja, solo donde vive cada dato):
//  1) BrLock: cerrojo de lectores sin escritura compartida (un contador por hilo lector en SU linea de 64 bytes; el
//     escritor pone su bandera y espera a que los contadores esten a 0: Dekker con barreras seq_cst). Sustituye al
//     std::shared_mutex del catalogo de tablas (RenderExecutor::m_catalog_mutex), que los trabajadores tomaban
//     compartido en CADA draw y dispatch (lock_shared + unlock_shared = 2 escrituras atomicas en la linea del
//     cerrojo, que era la misma que m_catalog/m_role/m_catalog_readers/m_table_xpr, leidas por el render en cada draw).
//  2) Contadores por hilo (bancos alineados a 64) de lo evitado, para la linea [tej] lineas.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace TejLinea {

// Lo evitado (o lo que queda) por tipo. Bancos por hilo: nadie comparte linea al contar.
enum C : uint32_t {
	CatRead,     // lecturas del catalogo de tablas sin escritura compartida (antes 2 atomicos en la linea del render)
	CatFallback, // lecturas con el cerrojo de reserva (mas hilos lectores que ranuras: debe ser 0)
	CatWriteWait,// el escritor (render) encontro lectores dentro
	TexTouch,    // GestorMem::Tocar de la sombra evitados (la sombra no toca la LRU: el render lo hace al dibujar)
	TouchSame,   // GestorMem::Tocar del render evitados (ya tenia ese fotograma: sin escritura)
	BcLookup,    // lookups de BufferCache de la sombra contados en su banco (antes fetch_add en una linea global)
	TrackLock,   // QUEDAN: cerrojos TrackingSpinLock (textura / regiones) tomados por la sombra
	NC
};
inline constexpr const char* Names[NC] = {"catalogo sin escritura", "catalogo reserva", "catalogo: render espero lectores",
                                          "Tocar sombra", "Tocar render repetido", "lookups buffer (banco)",
                                          "QUEDAN cerrojos de seguimiento de la sombra"};
struct alignas(64) Bank {
	std::atomic<uint64_t> c[NC] {};
};
inline constexpr uint32_t Banks = 32;
inline Bank               g_bank[Banks];
inline std::atomic<uint32_t> g_next_bank {0};
[[nodiscard]] inline uint32_t MyBank() {
	static thread_local const uint32_t b = g_next_bank.fetch_add(1, std::memory_order_relaxed) % Banks;
	return b;
}
// Hilo de la sombra (lector y trabajadores): lo pone tej.h, como TejLag::t_shadow (aqui sin depender de tej-lag.h).
inline thread_local bool t_shadow = false;
// Contar lo del render (TouchSame) solo con KYTY_TEJ y KYTY_TECHO=1 (lo pone tej.h): sin ellos ni un acceso de mas.
inline std::atomic<bool> g_count_touch {false};
inline void Add(C i, uint64_t v = 1) {
	auto& a = g_bank[MyBank()].c[i];
	a.store(a.load(std::memory_order_relaxed) + v, std::memory_order_relaxed); // (un solo escritor por banco... casi: ver Take)
}
// Suma y pone a 0 (hilo del informe). Bancos compartidos solo si hay mas de 32 hilos que cuentan: perder alguna cuenta
// ahi no importa (solo es informe).
inline void Take(uint64_t out[NC]) {
	for (uint32_t i = 0; i < NC; ++i) out[i] = 0;
	for (auto& b: g_bank)
		for (uint32_t i = 0; i < NC; ++i) out[i] += b.c[i].exchange(0, std::memory_order_relaxed);
}

// Cerrojo de lectores/escritor sin escrituras compartidas en la lectura (ver arriba). Interfaz de mutex compartido
// (std::shared_lock / std::unique_lock). Lectura anidada en el mismo hilo: permitida (no espera al escritor).
class BrLock {
public:
	static constexpr uint32_t Slots = 64;

	void lock_shared() {
		const int s = Slot();
		if (s < 0) {
			m_fallback.lock_shared();
			Add(CatFallback);
			return;
		}
		auto&          d   = m_slot[s].depth;
		const uint32_t cur = d.load(std::memory_order_relaxed);
		if (cur != 0) { // (anidada: ya dentro, el escritor nos espera a nosotros)
			d.store(cur + 1, std::memory_order_relaxed);
			return;
		}
		for (;;) {
			d.store(1, std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_seq_cst);
			if (m_writer.load(std::memory_order_relaxed) == 0) break;
			d.store(0, std::memory_order_release);
			for (uint32_t i = 0; m_writer.load(std::memory_order_acquire) != 0; ++i) {
				if (i < 256) _mm_pause();
				else std::this_thread::yield();
			}
		}
		std::atomic_thread_fence(std::memory_order_acquire);
		Add(CatRead);
	}
	void unlock_shared() {
		const int s = Slot();
		if (s < 0) {
			m_fallback.unlock_shared();
			return;
		}
		auto& d = m_slot[s].depth;
		d.store(d.load(std::memory_order_relaxed) - 1, std::memory_order_release);
	}
	void lock() {
		m_wmutex.lock();
		m_fallback.lock();
		m_writer.store(1, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_seq_cst);
		bool waited = false;
		for (uint32_t s = 0; s < Slots; ++s)
			for (uint32_t i = 0; m_slot[s].depth.load(std::memory_order_acquire) != 0; ++i) {
				waited = true;
				if (i < 256) _mm_pause();
				else std::this_thread::yield();
			}
		if (waited) Add(CatWriteWait);
	}
	bool try_lock() {
		if (!m_wmutex.try_lock()) return false;
		if (!m_fallback.try_lock()) {
			m_wmutex.unlock();
			return false;
		}
		m_writer.store(1, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_seq_cst);
		for (uint32_t s = 0; s < Slots; ++s)
			if (m_slot[s].depth.load(std::memory_order_acquire) != 0) {
				m_writer.store(0, std::memory_order_release);
				m_fallback.unlock();
				m_wmutex.unlock();
				return false;
			}
		return true;
	}
	void unlock() {
		m_writer.store(0, std::memory_order_release);
		m_fallback.unlock();
		m_wmutex.unlock();
	}

private:
	// Ranura del hilo lector (la misma en todos los BrLock: cada uno tiene las suyas). -1: sin ranura (reserva).
	static int Slot() {
		static std::atomic<uint32_t> next {0};
		static thread_local const int s = [] {
			const uint32_t n = next.fetch_add(1, std::memory_order_relaxed);
			if (n >= Slots) std::printf("[tej] lineas: hilo lector %u sin ranura en BrLock (cerrojo de reserva)\n", n);
			return n < Slots ? static_cast<int>(n) : -1;
		}();
		return s;
	}
	struct alignas(64) ReaderSlot {
		std::atomic<uint32_t> depth {0};
	};
	alignas(64) std::atomic<uint32_t> m_writer {0}; // (solo el escritor lo escribe; los lectores lo leen)
	alignas(64) std::mutex m_wmutex;                // (solo escritores)
	std::shared_mutex      m_fallback;              // (solo escritores y lectores sin ranura)
	ReaderSlot             m_slot[Slots];
};

inline void Report(double frames) {
	if (frames <= 0) return;
	uint64_t v[NC];
	Take(v);
	std::printf("[tej] lineas compartidas con el render, por fot (escrituras evitadas):");
	for (uint32_t i = 0; i < NC; ++i) std::printf("%s %s %.0f", i == 0 ? "" : ",", Names[i], static_cast<double>(v[i]) / frames);
	std::printf("\n");
}

} // namespace TejLinea
