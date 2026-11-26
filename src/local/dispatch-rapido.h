#pragma once
// (proto-B) KYTY_DISPATCH_RAPIDO=1 (apagado por defecto; apagado = todo igual que antes).
//
// Que hace (solo dispatches de calculo, hilo de render):
//  1. Copias lineales del juego (eboot 0x912e60: un dispatch por copia con dos V# y un CB {count, period}; Ghidra)
//     en lote: enciende el camino de copia1 (varias copias anunciadas seguidas = una comprobacion, un lock, un
//     PopPendingOperations), sin reordenar nada y sin pasarlas a la GPU (callejon: el juego reescribe constantes sin
//     etiqueta -> fallos de coincidencia). Necesita KYTY_DS_LINEAR_COPY_HOOK=1 (encendido en el lanzador).
//  2. Rescate de la via de tablas: un dispatch cuyo conjunto de imagenes tiene una textura con subida pendiente ya no
//     cae al camino normal (~38 us/dispatch en el pico: programa + rebind_buffers + rebind_images) sino que hace la
//     misma subida que haria el camino normal (FindTexture) y sigue por tablas (~2,8 us). La parte estable del
//     dispatch (par, programa, pipeline, conjunto de descriptores) es la que ya guarda la via de tablas; aqui solo se
//     evita perderla por una subida.
//  3. Censo del camino normal: cuantos dispatches caen ahi por cada motivo y cuanto cuestan (para decidir el
//     siguiente paso con datos, no a ciegas).
//
// Linea "[dispatch-rapido]" cada 5 s.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace DispatchRapido {

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_DISPATCH_RAPIDO");
		const bool  v    = text != nullptr && text[0] == '1' && text[1] == '\0';
		std::printf("[dispatch-rapido] %s (KYTY_DISPATCH_RAPIDO=%s)\n", v ? "ENCENDIDO" : "apagado", text != nullptr ? text : "");
		return v;
	}();
	return on;
}

[[nodiscard]] inline uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

constexpr uint32_t MaxReasons = 24;
struct Stats {
	// Camino normal, por motivo (puntero de texto estatico de m_dispatch_declined).
	std::array<const char*, MaxReasons> reason {};
	std::array<uint64_t, MaxReasons>    reason_count {}, reason_ns {};
	uint64_t normal = 0, normal_ns = 0;
	// Via de tablas: dispatches tomados y su tiempo; los rescatados (punto 2) y su tiempo.
	uint64_t table = 0, table_ns = 0, rescued = 0, rescued_ns = 0;
	// Copias lineales en lote (punto 1).
	uint64_t copies = 0, copy_runs = 0;
	uint64_t start = 0;
};
inline Stats g_stats; // (hilo de render)
inline thread_local bool g_rescued_now = false;
inline std::atomic<uint64_t> g_frames {0};
inline void NoteFrame() {
	if (On()) g_frames.fetch_add(1, std::memory_order_relaxed);
}

inline void Report() {
	auto&          s   = g_stats;
	const uint64_t now = NowNs();
	if (s.start == 0) {
		s.start = now;
		g_frames.store(0, std::memory_order_relaxed);
		return;
	}
	if (now - s.start < 5'000'000'000ull) return;
	const double f      = std::max<double>(1.0, static_cast<double>(g_frames.exchange(0, std::memory_order_relaxed)));
	const double normal = s.normal != 0 ? static_cast<double>(s.normal_ns) / static_cast<double>(s.normal) : 0.0;
	const double resc   = s.rescued != 0 ? static_cast<double>(s.rescued_ns) / static_cast<double>(s.rescued) : 0.0;
	// Ahorro: cada rescatado habria costado lo que cuesta de media un dispatch del camino normal en esta ventana.
	const double saved_ms = s.rescued * std::max(0.0, normal - resc) / 1e6 / f;
	std::printf("[dispatch-rapido] 5 s, %.0f frames: via rapida %.1f/frame (tablas %.1f a %.2f us, rescatados %.1f a %.2f us; "
	            "copias lineales en lote %.1f en %.1f lotes) | ahorrado ~%.2f ms/frame | camino normal %.1f/frame a %.2f us "
	            "(%.2f ms/frame), motivos:",
	            f, (s.table + s.copies) / f, s.table / f, s.table != 0 ? s.table_ns / 1e3 / s.table : 0.0, s.rescued / f,
	            resc / 1e3, s.copies / f, s.copy_runs / f, saved_ms, s.normal / f, normal / 1e3, s.normal_ns / 1e6 / f);
	for (uint32_t i = 0; i < MaxReasons && s.reason[i] != nullptr; ++i)
		std::printf(" \"%s\"=%.1f (%.2f ms)", s.reason[i], s.reason_count[i] / f, s.reason_ns[i] / 1e6 / f);
	std::printf("\n");
	s       = {};
	s.start = now;
}

inline void NoteNormal(const char* why, uint64_t ns) {
	auto& s = g_stats;
	++s.normal;
	s.normal_ns += ns;
	if (why == nullptr) why = "(sin motivo: via de tablas no probada)";
	for (uint32_t i = 0; i < MaxReasons; ++i) {
		if (s.reason[i] == nullptr) s.reason[i] = why;
		if (s.reason[i] == why || std::strcmp(s.reason[i], why) == 0) {
			++s.reason_count[i];
			s.reason_ns[i] += ns;
			break;
		}
	}
	Report();
}

inline void NoteTable(uint64_t ns) {
	auto& s = g_stats;
	++s.table;
	s.table_ns += ns;
	if (g_rescued_now) {
		++s.rescued;
		s.rescued_ns += ns;
	}
	g_rescued_now = false;
	Report();
}

inline void NoteCopies(uint32_t copies) {
	g_stats.copies += copies;
	++g_stats.copy_runs;
}

// Medicion de un dispatch del camino normal (desde que la via de tablas lo rechaza hasta que vuelve).
struct NormalTimer {
	const char* const* why = nullptr;
	uint64_t           t0  = 0;
	void Start(const char* const* reason) {
		if (!On()) return;
		why = reason;
		t0  = NowNs();
	}
	void Cancel() { why = nullptr; }
	~NormalTimer() {
		if (why != nullptr) NoteNormal(*why, NowNs() - t0);
	}
};

} // namespace DispatchRapido
