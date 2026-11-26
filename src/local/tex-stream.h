#pragma once
// (TEX-STREAM, 11-25) KYTY_TEX_STREAM=1 (por defecto; =0 lo apaga). Mips bajo demanda CON EL PROPIO JUEGO.
//
// Ghidra (eboot PPSA01341 01.007.000, base 0; en ejecucion eboot en 0x900000000):
//  - 0xac5b70 FUN_00ac5b70(req, ?, mip_pedido, uso): calcula el trozo deseado de una textura.
//    Peticion (0x28 B): +0 int trozo actual (n trozos residentes; MAS = mas detalle), +8 CTexture*, +0x12 u8 deseado,
//    +0x13 u8 maximo (todos los trozos), +0x14 u8 minimo, +0x18 (tex+0x280)<<24|uso, +0x1c texeles de mas
//    (bit31 = no muestreada este fotograma), +0x20 fotogramas con el mismo deseado.
//    tex+0x6c > 0 (fijada) => deseado = maximo. Al final: +0x20 = (deseado igual) ? +1 : 1.
//    Prologo (14 B, sin rip-relativo): 55 48 89 e5 41 57 41 56 41 55 41 54 53 50 | 48 8b 05 ... (0xac5b7e).
//  - 0xac5da0 aplica el deseado: baja con op 2 (decommit 0x9014f4) o sube trozo a trozo (0x901330 i+1).
//  - Pasada 1 de 0xac9d80: suelta trozos de las peticiones con actual > deseado (pool del juego ya a 0: siempre).
// Aqui: 0xac5b70 se desvia (jmp absoluto de 14 B, escrito con cmpxchg16b; la funcion esta alineada a 16) a Gancho():
// corre el original (trampolin con el prologo copiado) y despues limita el deseado segun la presion de VRAM:
//   (m9) nivel 1 (>= 92 %): no muestreadas a su minimo. Nivel 2 (>= 100 %): lejanas (lejania >= 3) -1 trozo.
//   Nivel 3 (>= 103 %): lejania >= 2, -1 trozo. < 97 % baja un nivel cada 200 ms; < 90 % nivel 0. "inutil": ver Tick. Las fijadas (tex+0x6c > 0) nunca se tocan.
// El juego suelta los trozos (decommit -> la VRAM baja) y los vuelve a pedir cuando el nivel baja: nunca se pierde
// contenido (la fuente es el archivo del juego), nada de texturas negras.
//
// Linea [tex-stream] cada 5 s (la escribe textureCache.cpp: TexStream::Report).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <intrin.h>
#endif
#include "gestor-memoria.h" // (M7) el nivel lo decide la presion de VRAM del gestor central

#if defined(_WIN32) && defined(__clang__)
#define KYTY_TEXSTREAM_OK 1
#define KYTY_TEXSTREAM_SYSV __attribute__((sysv_abi))
extern "C" __declspec(dllimport) void* __stdcall VirtualAlloc(void*, unsigned long long, unsigned long, unsigned long);
extern "C" __declspec(dllimport) int __stdcall VirtualProtect(void*, unsigned long long, unsigned long, unsigned long*);
extern "C" __declspec(dllimport) int __stdcall FlushInstructionCache(void*, const void*, unsigned long long);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
extern "C" __declspec(dllimport) int __stdcall IsBadReadPtr(const void*, unsigned long long);
#else
#define KYTY_TEXSTREAM_OK 0
#define KYTY_TEXSTREAM_SYSV
#endif

namespace TexStream {

constexpr uint64_t kEbootBase = 0x900000000ull;
constexpr uint64_t kDeseadoRva = 0xac5b70;
constexpr uint8_t  kPrologo[16] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                   0x41, 0x55, 0x41, 0x54, 0x53, 0x50, 0x48, 0x8b};

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TEX_STREAM");
		const bool  r = v != nullptr && v[0] == '1'; // (11-25) apagado por defecto: calidad completa
		std::printf("[tex-stream] KYTY_TEX_STREAM=%d: mips bajo demanda con el streamer del juego (0xac5b70)\n", r ? 1 : 0);
		std::fflush(stdout);
		return r;
	}();
	return on;
}

// Contadores (acumulados; Report saca la diferencia de la ventana).
inline std::atomic<int>      g_nivel {0};
inline std::atomic<bool>     g_instalado {false};
inline std::atomic<uint64_t> c_llamadas {0}, c_limitadas {0}, c_trozos_limitados {0};
inline std::atomic<uint64_t> c_soltados {0}, c_repedidos {0};
inline std::atomic<uint64_t> c_protegidas {0}, c_protegidas_bytes {0}; // GC: sin RAM (VRAM_UNICA) -> no se expulsan
inline std::atomic<uint64_t> c_dedup_n {0}, c_dedup_bytes {0};       // contenido igual en otra direccion (indice)

// Ultimo trozo actual visto por peticion (tabla abierta sin bloqueo; una carrera solo pierde una cuenta).
struct Visto {
	std::atomic<uint64_t> req {0};
	std::atomic<int32_t>  actual {0};
};
inline Visto g_vistos[1u << 16u];

using FnDeseado = void(KYTY_TEXSTREAM_SYSV*)(uint8_t*, uint64_t, uint32_t, uint32_t);
inline FnDeseado g_original = nullptr;

inline void NotaActual(uint8_t* req, int32_t actual) {
	const uint64_t key = reinterpret_cast<uint64_t>(req);
	uint32_t       h   = static_cast<uint32_t>((key >> 3u) * 0x9E3779B1u) >> 16u;
	for (uint32_t i = 0; i < 8; i++, h = (h + 1u) & 0xffffu) {
		auto&    v = g_vistos[h];
		uint64_t k = v.req.load(std::memory_order_relaxed);
		if (k == 0 && v.req.compare_exchange_strong(k, key, std::memory_order_relaxed)) {
			v.actual.store(actual, std::memory_order_relaxed);
			return;
		}
		if (k != key) continue;
		const int32_t antes = v.actual.exchange(actual, std::memory_order_relaxed);
		if (actual < antes) c_soltados.fetch_add(static_cast<uint64_t>(antes - actual), std::memory_order_relaxed);
		else if (actual > antes) c_repedidos.fetch_add(static_cast<uint64_t>(actual - antes), std::memory_order_relaxed);
		return;
	}
}

// (m8) LOD por distancia, como un PC: el trozo deseado que calcula el juego (0xac5b70: sale del mip que pide la GPU, es
// decir del tamano en pantalla = distancia) mide lo lejos que esta: lejania = maximo - deseado (0 = cerca, a todo detalle).
//  - Cercanas (lejania < g_umbral): NUNCA se tocan, a calidad completa.
//  - Lejanas (lejania >= g_umbral): -1 trozo (nivel 2) o -2 (nivel 3), nunca bajo el minimo del juego.
//  - No muestreadas este fotograma (bit31 de +0x1c): a su minimo desde el nivel 1.
// El umbral (en trozos) se acerca solo si la VRAM pasa del tope (3 -> 2 -> 1) y se aleja al haber sitio.
inline std::atomic<int>      g_umbral {3};
inline std::atomic<uint64_t> c_visibles_limitadas {0}, c_no_muestreadas_limitadas {0};

inline void KYTY_TEXSTREAM_SYSV Gancho(uint8_t* req, uint64_t a, uint32_t b, uint32_t c) {
	const uint8_t  prev     = req[0x12];
	const uint32_t prev_cnt = *reinterpret_cast<uint32_t*>(req + 0x20);
	g_original(req, a, b, c);
	c_llamadas.fetch_add(1, std::memory_order_relaxed);
	NotaActual(req, *reinterpret_cast<int32_t*>(req));
	const int nivel = g_nivel.load(std::memory_order_relaxed);
	if (nivel <= 0) return;
	const uint8_t* tex = *reinterpret_cast<uint8_t* const*>(req + 8);
	if (tex == nullptr || *reinterpret_cast<const int32_t*>(tex + 0x6c) > 0) return; // fijada
	const int  want = req[0x12], mx = req[0x13], mn = req[0x14];
	const bool no_muestreada = (*reinterpret_cast<const uint32_t*>(req + 0x1c) & 0x80000000u) != 0;
	int        cap = want;
	if (no_muestreada) {
		cap = mn;
	} else {
		const int lejania = mx - want;
		int umbral = g_umbral.load(std::memory_order_relaxed);
		if (umbral < 2) umbral = 2; // (m9) cercanas y medianas siempre a calidad completa
		if (nivel < 2 || lejania < umbral) return;
		cap = want - 1; // (m9) como maximo 1 trozo menos a las lejanas
	}
	if (cap < mn) cap = mn;
	if (cap >= want) return;
	req[0x12] = static_cast<uint8_t>(cap);
	*reinterpret_cast<uint32_t*>(req + 0x20) = static_cast<uint8_t>(cap) == prev ? prev_cnt + 1 : 1;
	c_limitadas.fetch_add(1, std::memory_order_relaxed);
	c_trozos_limitados.fetch_add(static_cast<uint64_t>(want - cap), std::memory_order_relaxed);
	(no_muestreada ? c_no_muestreadas_limitadas : c_visibles_limitadas).fetch_add(1, std::memory_order_relaxed);
}

// Hilo de render, una vez (cuando el eboot ya esta cargado). Devuelve si quedo puesto.
inline bool Instalar() {
#if KYTY_TEXSTREAM_OK
	static bool probado = false;
	if (probado) return g_instalado.load();
	probado = true;
	auto* at = reinterpret_cast<uint8_t*>(kEbootBase + kDeseadoRva);
	if (IsBadReadPtr(at, 16) != 0 || std::memcmp(at, kPrologo, 16) != 0) {
		std::printf("[tex-stream] 0xac5b70 no es el prologo esperado (eboot distinto de 01.007.000): apagado\n");
		std::fflush(stdout);
		return false;
	}
	auto* tramp = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, 0x3000 /*COMMIT|RESERVE*/, 0x40 /*RWX*/));
	if (tramp == nullptr) return false;
	std::memcpy(tramp, at, 14); // prologo (sin rip-relativo)
	tramp[14] = 0xff, tramp[15] = 0x25;
	std::memset(tramp + 16, 0, 4);
	const uint64_t vuelta = kEbootBase + kDeseadoRva + 14;
	std::memcpy(tramp + 20, &vuelta, 8);
	FlushInstructionCache(GetCurrentProcess(), tramp, 28);
	g_original = reinterpret_cast<FnDeseado>(tramp);

	alignas(16) uint8_t nuevo[16];
	nuevo[0] = 0xff, nuevo[1] = 0x25;
	std::memset(nuevo + 2, 0, 4);
	const uint64_t destino = reinterpret_cast<uint64_t>(&Gancho);
	std::memcpy(nuevo + 6, &destino, 8);
	nuevo[14] = at[14], nuevo[15] = at[15];
	unsigned long viejo = 0;
	if (VirtualProtect(at, 16, 0x40, &viejo) == 0) return false;
	alignas(16) long long esperado[2];
	std::memcpy(esperado, at, 16);
	long long nv[2];
	std::memcpy(nv, nuevo, 16);
	// cmpxchg16b: los 16 B cambian de una vez (otro hilo nunca ve medio salto).
	const bool ok = _InterlockedCompareExchange128(reinterpret_cast<volatile long long*>(at), nv[1], nv[0], esperado) != 0;
	unsigned long tmp = 0;
	VirtualProtect(at, 16, viejo, &tmp);
	FlushInstructionCache(GetCurrentProcess(), at, 16);
	g_instalado.store(ok);
	std::printf("[tex-stream] gancho en eboot+0xac5b70 %s\n", ok ? "PUESTO" : "NO puesto (cmpxchg16b fallo)");
	std::fflush(stdout);
	return ok;
#else
	return false;
#endif
}

// Hilo de render, cada 200 ms. used/tope en bytes (tope = tope efectivo de vram-tope o presupuesto del driver).
// (m9) Sin estados pegados: nivel 2 (>= 100 %, umbral 3) / 3 (>= 103 %, umbral 2), siempre -1 trozo a las lejanas;
// por debajo de 97 % baja un nivel por segundo y el umbral vuelve a 3; < 90 % nivel 0. Si en 10 s limitando el juego
// casi no suelta trozos y la VRAM no baja, el limite es INUTIL (el exceso no son texturas): nivel 0 durante 60 s.
inline std::atomic<bool> g_inutil {false};
inline void Tick(uint64_t used, uint64_t tope) {
	if (!On() || tope == 0) return;
	if (!Instalar()) return;
	uint64_t pct = used * 100 / tope;
	if (GestorMem::On() && GestorMem::Presupuesto(GestorMem::Vram) != 0) // (m9) sin forzar 100 % por "critica": solo uso/tope
		pct = static_cast<uint64_t>(GestorMem::Presion(GestorMem::Vram) * 100.0);
	// (m9) llamado cada 200 ms (textureCache.cpp): tiempos en ms reales, no en llamadas.
	const uint64_t ahora = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                           std::chrono::steady_clock::now().time_since_epoch()).count());
	static uint64_t espera_hasta = 0, ventana_desde = 0;
	static uint64_t v_sol = 0, v_used = 0;
	int n = g_nivel.load(std::memory_order_relaxed);
	int u = g_umbral.load(std::memory_order_relaxed);
	if (espera_hasta != 0) { // tras "inutil": 60 s sin limitar visibles
		if (ahora < espera_hasta) {
			g_nivel.store(pct >= 92 ? 1 : 0, std::memory_order_relaxed);
			g_umbral.store(3, std::memory_order_relaxed);
			return;
		}
		espera_hasta = 0;
		g_inutil.store(false, std::memory_order_relaxed);
	}
	if (pct >= 100) {
		n = pct >= 103 ? 3 : 2;
		u = n >= 3 ? 2 : 3;
	} else if (pct >= 97) {
		// banda de histeresis: se queda como esta (sin subir)
	} else if (pct >= 90) {
		n = n > 1 ? n - 1 : (pct >= 92 ? 1 : 0);
		u = 3;
	} else {
		n = 0, u = 3;
	}
	if (u < 2) u = 2;
	// Utilidad: ventana de 10 s con visibles limitadas.
	if (n >= 2) {
		const uint64_t sol = c_soltados.load(std::memory_order_relaxed);
		if (ventana_desde == 0) ventana_desde = ahora, v_sol = sol, v_used = used;
		if (ahora - ventana_desde >= 10000) {
			const bool suelta = sol - v_sol >= 64;
			const bool baja   = used + (64ull << 20u) <= v_used;
			ventana_desde     = 0;
			if (!suelta && !baja) {
				g_inutil.store(true, std::memory_order_relaxed);
				espera_hasta = ahora + 60000, n = pct >= 92 ? 1 : 0, u = 3;
			}
		}
	} else {
		ventana_desde = 0;
	}
	g_umbral.store(u, std::memory_order_relaxed);
	g_nivel.store(n, std::memory_order_relaxed);
}

// (M7) Antes se registraba un liberador que subia el nivel por su cuenta: ahora el nivel sale solo de
// GestorMem::Presion(Vram) en Tick() (un solo arbitro, sin dos umbrales peleando). Se deja vacio por compatibilidad.
inline void RegistrarEnGestor() {}

// Hilo de render: linea cada 5 s.
inline void Report(uint64_t now_ms, uint64_t tex_used, uint64_t dev_used, uint64_t tope, uint64_t expulsadas,
                   uint64_t expulsadas_bytes, uint64_t resubidas, uint64_t resubidas_bytes) {
	if (!On()) return;
	static uint64_t last = 0, l_lim = 0, l_tr = 0, l_sol = 0, l_rep = 0, l_ev = 0, l_evb = 0, l_ru = 0, l_rub = 0;
	if (last == 0) last = now_ms;
	if (now_ms - last < 5000) return;
	last              = now_ms;
	const auto lim = c_limitadas.load(), tr = c_trozos_limitados.load(), sol = c_soltados.load(), rep = c_repedidos.load();
	std::printf("[tex-stream] 5 s: VRAM %llu MiB (texturas %llu; tope %llu), nivel %d%s (umbral lejania %d trozos), gancho %s; mips limitados %llu "
	            "peticiones (%llu trozos; visibles %llu, no muestreadas %llu); trozos soltados por el juego %llu, vueltos a pedir %llu; dedup real %llu imagenes "
	            "compartiendo VkImage (%llu MiB ahorrados ahora); expulsadas %llu (%llu MiB), resubidas %llu (%llu MiB); "
	            "sin expulsar por estar solo en VRAM %llu (%llu MiB)\n",
	            static_cast<unsigned long long>(dev_used >> 20u), static_cast<unsigned long long>(tex_used >> 20u),
	            static_cast<unsigned long long>(tope >> 20u), g_nivel.load(), g_inutil.load() ? " inutil" : "", g_umbral.load(), g_instalado.load() ? "puesto" : "NO",
	            static_cast<unsigned long long>(lim - l_lim), static_cast<unsigned long long>(tr - l_tr),
	            static_cast<unsigned long long>(c_visibles_limitadas.exchange(0)), static_cast<unsigned long long>(c_no_muestreadas_limitadas.exchange(0)),
	            static_cast<unsigned long long>(sol - l_sol), static_cast<unsigned long long>(rep - l_rep),
	            static_cast<unsigned long long>(c_dedup_n.load()), static_cast<unsigned long long>(c_dedup_bytes.load() >> 20u),
	            static_cast<unsigned long long>(expulsadas - l_ev), static_cast<unsigned long long>((expulsadas_bytes - l_evb) >> 20u),
	            static_cast<unsigned long long>(resubidas - l_ru), static_cast<unsigned long long>((resubidas_bytes - l_rub) >> 20u),
	            static_cast<unsigned long long>(c_protegidas.load()),
	            static_cast<unsigned long long>(c_protegidas_bytes.load() >> 20u));
	std::fflush(stdout);
	l_lim = lim, l_tr = tr, l_sol = sol, l_rep = rep, l_ev = expulsadas, l_evb = expulsadas_bytes, l_ru = resubidas,
	l_rub = resubidas_bytes;
	c_protegidas.store(0), c_protegidas_bytes.store(0);
}

} // namespace TexStream
