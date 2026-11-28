#pragma once
// (tej-coser etapa 1, KYTY_TEJ=3) Sonda de llamadas Vulkan por hilo: hash de la secuencia de comandos de TRABAJO
// (draw, dispatch, copias, limpiezas, barreras, comienzos/fines de rendering, ExecuteCommands) con sus argumentos
// escalares de 32 bits (cuentas, primeros indices, desplazamientos de vertice, dimensiones de dispatch...). Sin handles
// ni VkDeviceSize: el trabajador usa sus propios anillos/sets, la secuencia de trabajo y sus cuentas deben ser iguales.
// Los binds y el estado dinamico no entran (el render los reutiliza/elide de otra forma: no cambian lo que se dibuja si
// la secuencia de trabajo y el estado de registros coinciden, eso ya lo valida F1d).
// Coste apagado: una lectura de un thread_local por comando (t_on). Solo se enciende en paquetes que se miden.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>

namespace TejCoser {

// Clases (las pone el generador de vulkan-recording por nombre).
enum Cls : uint32_t {
	CDraw = 1,     // draw / ClearAttachments (validos dentro de un rendering, en un secundario)
	CCopy = 2,     // copias, rellenos, limpiezas de imagen, blit, resolve, UpdateBuffer: NO validos dentro de un rendering
	CBarrier = 3,  // barreras: no se permiten dentro del paquete secundario
	CBegin = 4,    // vkCmdBeginRendering
	CEnd = 5,      // vkCmdEndRendering
	CExec = 6,     // vkCmdExecuteCommands
	CStamp = 7,    // timestamps (diagnostico: fuera del hash)
	CDispatch = 8, // dispatch: no valido dentro de un rendering
};

struct Probe {
	uint64_t hash = 0xcbf29ce484222325ull;
	uint32_t draws = 0, copies = 0, barriers = 0, begins = 0, ends = 0, execs = 0, dispatches = 0, calls = 0;
	// Valido como secundario DENTRO del rendering que hereda: solo draws (y ClearAttachments).
	[[nodiscard]] bool SecondaryOk() const noexcept {
		return copies == 0 && barriers == 0 && begins == 0 && ends == 0 && execs == 0 && dispatches == 0;
	}
};

inline thread_local bool  t_on = false;
inline thread_local Probe t_probe;
inline std::atomic<bool>  g_installed {false}; // (la sonda solo existe con KYTY_LOCAL_VULKAN_RECORDING: Install)

[[nodiscard]] constexpr uint32_t Id(const char* s) noexcept {
	uint32_t h = 2166136261u;
	for (; *s != 0; ++s) h = (h ^ static_cast<uint8_t>(*s)) * 16777619u;
	return h;
}
[[nodiscard]] inline uint64_t Mix(uint64_t h, uint64_t v) noexcept {
	h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
	return h * 0x100000001b3ull;
}

[[gnu::noinline]] inline void Call(uint32_t id, uint32_t cls, std::initializer_list<uint64_t> scalars) noexcept {
	auto& p = t_probe;
	++p.calls;
	switch (cls) {
		case CDraw: ++p.draws; break;
		case CCopy: ++p.copies; break;
		case CBarrier: ++p.barriers; break;
		case CBegin: ++p.begins; break;
		case CEnd: ++p.ends; break;
		case CExec: ++p.execs; break;
		case CDispatch: ++p.dispatches; break;
		default: return; // (CStamp: fuera del hash)
	}
	p.hash = Mix(p.hash, id);
	for (const auto v: scalars) p.hash = Mix(p.hash, v);
}

inline void Start() noexcept {
	t_probe = {};
	t_on    = g_installed.load(std::memory_order_relaxed);
}
[[nodiscard]] inline Probe Stop() noexcept {
	t_on = false;
	return t_probe;
}

// KYTY_TEJ=3 (lo lee tambien vulkanWindow.cpp para maintenance7, antes que tej.h).
[[nodiscard]] inline bool Wanted() noexcept {
	static const bool w = [] {
		const char* v = std::getenv("KYTY_TEJ");
		return v != nullptr && std::atoi(v) == 3;
	}();
	return w;
}
// maintenance7 (renderingContentsInline: draws propios y secundarios en el mismo rendering) activado en el dispositivo.
inline std::atomic<bool> g_m7 {false};

} // namespace TejCoser
