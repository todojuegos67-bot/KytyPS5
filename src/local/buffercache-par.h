#pragma once
// (buffercache-par) KYTY_BC_PAR (apagado por defecto): BufferCache consultable sin rechazo desde un hilo de traduccion
// especulativa (KYTY_SPECULATE=4, docs/PARALLEL-TRANSLATION.md §10: ~170 huecos/fot en frio en el hilo GPU).
//
// 1) Busqueda concurrente segura. Lo unico del indice que no era seguro leer desde otro hilo es la tabla de paginas
//    (MultiLevelPageTable: cubo de 16 MiB creado con un unique_ptr no atomico, entradas BufferId de 8 bytes escritas sin
//    atomicidad -> lectura partida index/generation). SlotVector ya es seguro (trozos publicados atomicos, vida atomica)
//    y el borrado se difiere (EraseRetired + Spec::PacketsPassed). Por eso: 64 fragmentos shared_mutex por granulo de
//    16 MiB (= un cubo de la tabla: el puntero del cubo y sus entradas quedan bajo el mismo fragmento). El hilo GPU
//    (unico escritor) toma en exclusiva solo los fragmentos que cubre el buffer al registrarlo / quitarlo
//    (ChangeRegister); los hilos de traduccion toman compartido el del vaddr que buscan. El hilo GPU no se bloquea a
//    si mismo para leer (es el escritor).
// 2) "memory the processor writes": en vez de hueco/parada, el segmento abierto se cierra justo antes del paquete
//    (corte sin paquete: el Commit aplica en orden las escrituras de la DCB de los segmentos anteriores antes del
//    trabajo del siguiente) y el paquete se traduce otra vez en el segmento nuevo. Una sola vez por paquete.
// 3) Registro de buffers nuevos desde el hilo secundario: SIGUE rechazado (contado). No se encola: el hueco que deja
//    lo traduce el hilo GPU en orden y ese mismo hueco ya registra el buffer (la siguiente vez se encuentra); un
//    descriptor provisional exigiria reescribir comandos ya grabados (VkBuffer en vkCmdBind*) -> seguridad < 90 %.
//
// Modos: 0 apagado (identico a antes). 1 encendido. 2 A/B cada ventana de 5 s de [techo] (KYTY_TECHO=1): B = encendido;
// sin KYTY_SPECULATE=4 alterna tambien los cerrojos del escritor (mide su coste con un hilo); con =4 los cerrojos
// quedan siempre puestos (no se puede quitar un cerrojo con lectores vivos) y solo alterna el corte (2).
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <shared_mutex>

#include "local/techo.h"
#include "local/tej-lag.h"
#include "local/tej-linea.h"

extern "C" { extern volatile std::atomic_uint32_t kyty_local_speculate_mode; } // (fix) misma linkage C que speculation.h

namespace BcPar {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_BC_PAR");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[buffercache-par] KYTY_BC_PAR=2 sin KYTY_TECHO=1: modo 1\n");
			m = 1;
		}
		if (m != 0) std::printf("[buffercache-par] KYTY_BC_PAR=%d (%s)\n", m, m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};

[[nodiscard]] inline bool SpecThreads() { return kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4; }
// (tej, KYTY_TEJ=1/2) La sombra lee el indice desde su hilo: cerrojos de lector y escritor siempre puestos (no el corte).
[[nodiscard]] inline bool TejOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEJ");
		return v != nullptr && (std::atoi(v) == 1 || std::atoi(v) == 2);
	}();
	return on;
}
// Lectores (hilos de traduccion): fijo durante toda la ejecucion.
[[nodiscard]] inline bool ReadLocks() { return Mode() != 0 || TejOn(); }
// Escritor (hilo GPU).
[[nodiscard]] inline bool WriteLocks() {
	const int m = Mode();
	return m == 1 || (m == 2 && (SpecThreads() || g_ab_on.load(std::memory_order_relaxed))) || TejOn();
}
[[nodiscard]] inline bool SplitOn() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

inline constexpr uint32_t Shards = 64, GranuleBits = 24; // (16 MiB = un cubo de la tabla de paginas de BufferCache)
// (tej-linea) Un fragmento por linea de 64 bytes: antes 8 por linea (SRWLOCK de 8 bytes), y el lock_shared de un trabajador
// quitaba la linea a los demas fragmentos (los del render al registrar y los de los otros trabajadores).
struct alignas(64) Shard {
	std::shared_mutex m;
	void lock() { m.lock(); }
	bool try_lock() { return m.try_lock(); }
	void unlock() { m.unlock(); }
	void lock_shared() { m.lock_shared(); }
	void unlock_shared() { m.unlock_shared(); }
};
inline Shard              g_shard[Shards];
[[nodiscard]] inline uint32_t ShardOf(uint64_t vaddr) { return static_cast<uint32_t>(vaddr >> GranuleBits) & (Shards - 1); }

// Contadores (por fotograma en [buffercache-par]).
enum Why : uint32_t { RegNew, LookupMiss, LookupTexel, JournalRefused, Count };
// (tej-lag) Los del escritor (render) en su propia linea de cache: los lectores (sombra) no la tocan (false sharing).
alignas(64) inline std::atomic<uint64_t> g_write_locks {0};
inline std::atomic<uint64_t>             g_write_ns {0};
alignas(64) inline std::atomic<uint64_t> g_lookups {0};
inline std::atomic<uint64_t>             g_splits {0}, g_queued {0};
inline std::atomic<uint64_t> g_refused[Count] {};
inline thread_local bool     t_journal = false; // (el rechazo actual es "memory the processor writes" de BufferCache)
inline thread_local bool     t_reader  = false; // (hilo de traduccion especulativa: lee el indice sin ser el escritor)

inline void Refused(Why why) { g_refused[why].fetch_add(1, std::memory_order_relaxed); }

// Lectura de un hilo de traduccion (no GPU): compartido sobre el fragmento del vaddr.
struct ReadGuard {
	Shard* m = nullptr;
	explicit ReadGuard(uint64_t vaddr) {
		if (!t_reader || !ReadLocks()) return;
		m = &g_shard[ShardOf(vaddr)];
		m->lock_shared();
		TejLinea::Add(TejLinea::BcLookup); // (tej-linea: en el banco del hilo, antes fetch_add en una linea global)
	}
	~ReadGuard() {
		if (m != nullptr) m->unlock_shared();
	}
	ReadGuard(const ReadGuard&)            = delete;
	ReadGuard& operator=(const ReadGuard&) = delete;
};

// Escritura del hilo GPU sobre [begin, end): exclusivo en los fragmentos que toca, en orden creciente.
struct WriteGuard {
	uint64_t mask = 0;
	uint64_t t0   = 0;
	WriteGuard(uint64_t begin, uint64_t end) {
		if (!WriteLocks() || end <= begin) return;
		const uint64_t first = begin >> GranuleBits, last = (end - 1) >> GranuleBits;
		if (last - first + 1 >= Shards) mask = ~uint64_t {0};
		else
			for (uint64_t g = first; g <= last; ++g) mask |= uint64_t {1} << (g & (Shards - 1));
		t0 = Techo::On() ? __rdtsc() : 0;
		for (uint32_t i = 0; i < Shards; ++i)
			if (mask & (uint64_t {1} << i)) TejLag::Lock(g_shard[i], TejLag::BcWrite); // (tej-lag: espera del render)
	}
	~WriteGuard() {
		if (mask == 0) return;
		for (uint32_t i = 0; i < Shards; ++i)
			if (mask & (uint64_t {1} << i)) g_shard[i].unlock();
		g_write_locks.fetch_add(1, std::memory_order_relaxed);
		if (t0 != 0) g_write_ns.fetch_add(__rdtsc() - t0, std::memory_order_relaxed); // (ticks de TSC)
	}
	WriteGuard(const WriteGuard&)            = delete;
	WriteGuard& operator=(const WriteGuard&) = delete;
};

// Ventana de 5 s de [techo] (hilo del informe).
inline void Window(double busy_ms, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const bool was   = g_ab_on.load(std::memory_order_relaxed);
	double     delta = 0, cnt[2] {};
	if (m == 2) {
		struct W { bool b; double busy; };
		static W      w[4096];
		static size_t k = 0;
		if (keep && k < 4096) w[k++] = {was, busy_ms};
		double sum[2] {};
		for (size_t i = 1; i + 1 < k; ++i)
			if (w[i - 1].b != w[i].b && w[i + 1].b != w[i].b) {
				const double d = w[i].busy - 0.5 * (w[i - 1].busy + w[i + 1].busy);
				sum[w[i].b ? 1 : 0] += d, cnt[w[i].b ? 1 : 0] += 1;
			}
		const double db = cnt[1] > 0 ? sum[1] / cnt[1] : 0, da = cnt[0] > 0 ? sum[0] / cnt[0] : 0;
		delta = (cnt[0] > 0 && cnt[1] > 0) ? 0.5 * (db - da) : cnt[1] > 0 ? db : -da;
	}
	const auto per = [&](std::atomic<uint64_t>& a) { return static_cast<double>(a.exchange(0, std::memory_order_relaxed)) / frames; };
	const double lk = per(g_lookups), wl = per(g_write_locks), wt = per(g_write_ns), sp = per(g_splits), q = per(g_queued);
	const double r0 = per(g_refused[RegNew]), r1 = per(g_refused[LookupMiss]), r2 = per(g_refused[LookupTexel]),
	             r3 = per(g_refused[JournalRefused]);
	std::printf("[buffercache-par] ventana %s: render ocupado %.2f ms | por fot: lookups concurrentes %.0f, cerrojos escritor %.1f"
	            " (%.0f ticks TSC), cortes por escrituras de la DCB %.1f, registros encolados %.0f | rechazos que quedan:"
	            " registro %.1f, lookup sin buffer %.1f, lookup texel/imagen GPU %.1f, escrituras de la DCB %.1f",
	            m == 2 ? (was ? "B (encendido)" : "A (apagado)") : "encendido", busy_ms, lk, wl, wt, sp, q, r0, r1, r2, r3);
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (objetivo 0 +-0.2 con 1 hilo)", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace BcPar
