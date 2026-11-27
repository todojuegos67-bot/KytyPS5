#pragma once
// (do2-rapido) KYTY_DRAW_INDEX_OFFSET_2 fast path, KYTY_DO2_RAPIDO (apagado por defecto).
// Ghidra (eboot): AgcDcbDrawIndexOffset thunk 0x1d0b070, llamado desde FUN_00914360 / FUN_00914430 (sub-mallas:
// el puntero de user-data, FUN_01d108a0, solo se reescribe cuando cambia el modo param_2; llamadas seguidas con el
// mismo modo = solo DRAW_INDEX_OFFSET_2 con otro index_offset/index_count) y FUN_00adad90 (2 draws, modos 1 y 5, con
// estado distinto entre ellos: ahi NO aplica).
// Un DRAW_INDEX_OFFSET_2 / DRAW_INDEX_2 directo de la via de tablas cuyo draw anterior fue otro directo de tablas con
// el mismo par, variante, estado, set de descriptores, pipeline, user data identico, mismo command buffer/generacion/
// pase, mismas epocas (imagenes, registro de buffers, mapeos) y ningun paquete PM4 entre medias salvo registros
// (SET_SH/CONTEXT/UCONFIG, INDEX_*, NUM_INSTANCES, NOP): se dibuja con vkCmdDrawIndexed directo (mismos bloques y
// push constants del anterior; solo cambian indice base/cuenta/instancias y el offset de vertices), sin TableEvaluate,
// TableGather, TableSet, BDA ni bloques.
//   1 encendido.  2 A/B cada ventana de 5 s de [techo] (necesita KYTY_TECHO=1): B = encendido; delta interpolado.
// Apagado: una carga de un static por draw de tablas.
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"

namespace Do2Rapido {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_DO2_RAPIDO");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[do2-rapido] KYTY_DO2_RAPIDO=2 sin KYTY_TECHO=1: se usa 1 (siempre encendido)\n");
			m = 1;
		}
		if (m != 0) std::printf("[do2-rapido] KYTY_DO2_RAPIDO=%d (%s)\n", m, m == 1 ? "encendido" : "A/B cada 5 s, B = encendido");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false}; // (=2) esta ventana usa el atajo (B)

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// Paquetes PM4 vistos por el observador (DrawStateObserver::ObservePacket) que no son solo registros: draws,
// dispatches, escrituras a memoria, saltos, desconocidos. El atajo exige que entre los dos draws solo cuente el suyo.
inline thread_local uint64_t t_seq = 1;
[[nodiscard]] inline bool RegisterOnly(uint32_t opcode, uint32_t r_field);

// Lo que dejo el ultimo draw directo de tablas dibujado (via normal o atajo) en el hilo de render.
struct Snap {
	bool     valid = false;
	uint64_t seq = 0, frame = 0, shader_gen = 0;
	const void* pair = nullptr;
	const void* variant = nullptr;
	const void* state = nullptr;
	const void* programs[2] {};
	const void* set = nullptr;
	const void* pipeline = nullptr;
	uint64_t command = 0, generation = 0, passes = 0, images_epoch = 0, registration = 0, mapping = 0;
	uint32_t index_type = 0;
	uint32_t ud_size[2] {};
	std::array<std::array<uint32_t, 32>, 2> ud {};
	uint32_t                 push_dwords = 0;
	std::array<uint32_t, 32> push {};
	// El indice enlazado: buffer y su rango demostrado (como el HeldBuffer de obtain), base y bytes enlazados.
	void*    ib_buffer = nullptr;
	uint64_t ib_begin = 0, ib_end = 0, ib_cpu_epoch = 0, ib_registration = 0;
	uint64_t ib_base = 0, ib_bytes = 0;
};
inline Snap g_snap; // (solo el ejecutor principal, Role() == 0)

inline std::atomic<uint64_t> g_taken {0}, g_fallback {0}, g_would {0};
// Motivo del primer fallo de un directo no tomado (diagnostico por ventana).
enum Why : uint32_t { WNoSnap, WSeq, WKey, WUd, WCommand, WEpoch, WIndex, WExcluded, WCount };
inline constexpr const char* WhyNames[WCount] = {"sin-previo", "paquetes-entre", "par/estado", "user-data",
                                                 "cmd/pase", "epocas", "indice", "excluido"};
inline std::atomic<uint64_t> g_why[WCount] {};

inline bool RegisterOnly(uint32_t opcode, uint32_t r_field) {
	switch (opcode) {
		case 0x69: // IT_SET_CONTEXT_REG
		case 0x76: // IT_SET_SH_REG
		case 0x79: // IT_SET_UCONFIG_REG
		case 0x7a: // IT_SET_UCONFIG_REG_INDEX
		case 0x26: // IT_INDEX_BASE
		case 0x13: // IT_INDEX_BUFFER_SIZE
		case 0x2a: // IT_INDEX_TYPE
		case 0x2f: // IT_NUM_INSTANCES
			return true;
		case 0x10: // IT_NOP (no los que cambian contexto: el observador los invalida)
			return r_field == 0;
		default: return false;
	}
}

// Ventana de 5 s de [techo] (hilo del informe).
inline void Window(double busy_ms, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const double t = static_cast<double>(g_taken.exchange(0, std::memory_order_relaxed));
	const double f = static_cast<double>(g_fallback.exchange(0, std::memory_order_relaxed));
	const double w = static_cast<double>(g_would.exchange(0, std::memory_order_relaxed));
	double       why[WCount];
	for (uint32_t i = 0; i < WCount; ++i) why[i] = static_cast<double>(g_why[i].exchange(0, std::memory_order_relaxed));
	const bool was   = g_ab_on.load(std::memory_order_relaxed);
	double     delta = 0, cnt[2] {};
	if (m == 2) {
		struct W { bool b; double busy; };
		static W      win[4096];
		static size_t k = 0;
		if (keep && k < 4096) win[k++] = {was, busy_ms};
		double sum[2] {};
		for (size_t i = 1; i + 1 < k; ++i)
			if (win[i - 1].b != win[i].b && win[i + 1].b != win[i].b) {
				const double d = win[i].busy - 0.5 * (win[i - 1].busy + win[i + 1].busy);
				sum[win[i].b ? 1 : 0] += d, cnt[win[i].b ? 1 : 0] += 1;
			}
		const double db = cnt[1] > 0 ? sum[1] / cnt[1] : 0, da = cnt[0] > 0 ? sum[0] / cnt[0] : 0;
		delta = (cnt[0] > 0 && cnt[1] > 0) ? 0.5 * (db - da) : cnt[1] > 0 ? db : -da;
	}
	std::printf("[do2-rapido] ventana %s: render ocupado %.2f ms | atajo %.0f/fot, aptos sin usar (A) %.0f/fot, normal %.0f/fot (",
	            m == 2 ? (was ? "B (atajo)" : "A (normal)") : "atajo", busy_ms, t / frames, w / frames, f / frames);
	for (uint32_t i = 0; i < WCount; ++i) std::printf("%s%s %.0f", i ? ", " : "", WhyNames[i], why[i] / frames);
	std::printf(")");
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = ahorro)", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace Do2Rapido
