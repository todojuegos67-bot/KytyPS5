#ifndef EMULATOR_SRC_LOCAL_TRAD1_PRUEBA_H_
#define EMULATOR_SRC_LOCAL_TRAD1_PRUEBA_H_
// (trad1-prueba) KYTY_TRAD1_PRUEBA=1|2 (apagado por defecto). Los fallos del atajo trad1 (table-xpr.inc) por
// 'sin_prueba' y 'cobertura':
//  - sin_prueba: la entrada (direccion, tamano) no esta en la cache de pruebas (2 vias x 4096 conjuntos). Con
//    KYTY_TECHO=1 se separa con un historial de 65536 claves: 'nuevo' (rango nunca visto), 'desalojado' (estaba y la
//    cache de 2 vias lo echo: capacidad/conflicto) y 'tamano' (misma direccion, otro tamano: V# cambiado).
//    Mejora: cache de 4 vias x 16384 conjuntos (8x entradas) -> los 'desalojado' pasan a aciertos.
//  - cobertura: la prueba por region de 4 MiB es conservadora (una perdida en cualquier sitio de la region cuenta).
//    Mejora: si se movio la region, la prueba exacta del rango (todas sus paginas GPU-owned y el rango entero en
//    m_gpu_modified_ranges: lo que SynchronizeBuffer y el Add mirarian) decide; si vale, se reestampa la entrada.
//    Con TECHO: 'cobertura_exacta_ok' (falsos fallos recuperables) y 'cobertura_real'.
// =2: A/B cada 5 s (ventana comun de techo-report), B = encendido. Comprobacion: KYTY_TRAD1=2 sigue comparando el
// camino viejo con lo que el atajo habria tomado (DIFERENCIAS), tambien en las entradas que gana esta mejora.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "local/reent.h"
#include "local/techo.h"

namespace Trad1Prueba {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_TRAD1_PRUEBA");
		int         m = v != nullptr ? std::atoi(v) : 1; // (t69) encendido por defecto: medido sin fallos, DIFERENCIAS 0
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[trad1-prueba] KYTY_TRAD1_PRUEBA=2 sin KYTY_TECHO=1: queda en 1\n");
			m = 1;
		}
		if (m != 0) {
			std::printf("[trad1-prueba] KYTY_TRAD1_PRUEBA=%d (%s): cache de pruebas 4x16384 y cobertura exacta del rango\n", m,
			            m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
			std::fflush(stdout);
		}
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// Contadores (solo con KYTY_TECHO=1). (reentrante) Antes uint64_t globales sin atomica: ahora celdas por hilo (Reent),
// el informe suma los carriles.
// La cache de pruebas en si (TableXpr::trad1_ranges, 4x16384 x 56 B = 3.7 MB, table-xpr.inc) NO es global: es miembro
// de la TableXpr de cada RenderExecutor, y cada carril de par2 tiene sus executors (Par2::Lane::executors), asi que ya
// es por carril. Lo que esa cache prueba (registro, cobertura GPU, buffers vivos) vive en la BufferCache compartida:
// con varios carriles esas lecturas necesitan el bloqueo/diferido de la fase 2-3 de par2, no algo de este archivo.
struct Tag;
enum Cell : uint32_t { CNew, CEvicted, CSize, CCovExactOk, CCovReal, CCovWon, NCells };
inline void Count(Cell c) { Reent::Add<Tag, NCells>(c); }

struct Key {
	uint64_t address = 0, size = 0;
};
// Historial de claves (solo TECHO, 1 MiB): por hilo, para que dos carriles no lo creen/escriban a la vez. Con un hilo,
// el mismo historial de antes.
inline thread_local Key* t_hist = nullptr;
constexpr uint64_t       HistSize = 65536;
[[nodiscard]] inline Key* Hist() {
	if (t_hist == nullptr) t_hist = new Key[HistSize](); // vive hasta el final del proceso (un bloque por hilo)
	return t_hist;
}

[[nodiscard]] inline uint64_t HistSlot(uint64_t address) { return ((address >> 2u) * 0x9e3779b97f4a7c15ull) >> 48u; }

inline void NoteKept(uint64_t address, uint64_t size) {
	if (!Techo::On()) return;
	Hist()[HistSlot(address)] = {address, size};
}

inline void NoteNoProof(uint64_t address, uint64_t size) {
	if (!Techo::On()) return;
	const auto& k = Hist()[HistSlot(address)];
	Count(k.address != address ? CNew : k.size == size ? CEvicted : CSize);
}

inline void Window(double busy_ms, double frames, bool keep) {
	const int m = Mode();
	if (!Techo::On() || frames <= 0) return;
	uint64_t c[NCells];
	Reent::Take<Tag, NCells>(c);
	const bool was   = g_ab_on.load(std::memory_order_relaxed);
	double     delta = 0, cnt[2] {};
	if (m == 2) {
		struct W {
			bool   b;
			double busy;
		};
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
	if (c[CNew] + c[CEvicted] + c[CSize] + c[CCovExactOk] + c[CCovReal] + c[CCovWon] != 0 || m != 0) {
		std::printf("[trad1-prueba] ventana %s: render ocupado %.2f ms | sin_prueba nuevo %.1f/fot desalojado %.1f tamano %.1f | "
		            "cobertura exacta_ok %.1f real %.1f ganados %.1f",
		            m == 2 ? (was ? "B (encendido)" : "A (normal)") : m == 1 ? "encendido" : "apagado", busy_ms,
		            static_cast<double>(c[CNew]) / frames, static_cast<double>(c[CEvicted]) / frames,
		            static_cast<double>(c[CSize]) / frames, static_cast<double>(c[CCovExactOk]) / frames,
		            static_cast<double>(c[CCovReal]) / frames, static_cast<double>(c[CCovWon]) / frames);
		if (m == 2) std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot", cnt[0], cnt[1], delta);
		std::printf("\n");
		std::fflush(stdout);
	}
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace Trad1Prueba

#endif
