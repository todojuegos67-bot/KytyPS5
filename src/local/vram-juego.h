#pragma once
// (VRAM-JUEGO, 11-24) KYTY_VRAM_JUEGO=1 (por defecto; =0 lo apaga): la VRAM la baja EL PROPIO JUEGO, con su streaming
// de texturas, como en la PS5 (memoria compartida).
//
// Ghidra (eboot PPSA01341 01.007.000; eboot cargado en 0x900000000):
//  - 0xac6310 registra las opciones "Textures": "Streaming Pool Total Size MiBs" = objeto 0x2f69838 (valor: int MiB en
//    *[0x2f69898], objeto+0x60), "Low Memory Limit MiBs" 0x2f69938, "Low Memory Stop Streaming Offset MiBs" 0x2f69a38,
//    "Force Desired TopMip (0=Highest)" 0x2f69e38 (FUN_00824df0 guarda el puntero al valor en objeto+0x60).
//  - 0xac9ea0 = gestor del streaming (cada pasada). Lee el tamano del pool en 0xac9f50 (mov rax,[rip->0x2f69898]):
//      presupuesto = PoolMiB<<20 - base(no streaming, mgr+0x286048)   (si TopMip forzado < 0, el normal)
//      y si "Low Memory Limit" != 0: min(presupuesto, libre del asignador del juego 0x80d540 - limite - offset).
//    Lo guarda en mgr+0x286038. Si lo pedido pasa del presupuesto, ordena (0xacc070 modo 3) y baja el trozo deseado
//    (req+0x12) de las texturas hasta caber; si cabe, sube solo lo que entra.
//  - 0xac5da0/0xac6050 aplican el trozo: CTexture::SetCurrentStreamingChunk 0x901330. Bajar de trozo = op 2 sobre
//    0x7f9f70 (decommit) de los mips sobrantes y this+0x27c -= bytes; 0x901540 rehace el mip base del T#.
//  - Ese op 2 llega al emulador como unmap: la imagen nativa desliga el mip (sparse) y la VRAM baja al momento.
//  - El valor del pool solo lo leen 0xac9ea0 (presupuesto) y 0xac7a10 (texto de depuracion): no dimensiona ninguna
//    reserva de memoria. Es una variable de configuracion estatica (no memoria que el juego reutilice).
//
// (LEJANAS, 11-24) Ghidra del gestor (funcion real 0xac9d80, contiene 0xac9ea0/0xac9f50):
//  - Peticion (0x28 B): +0 trozo actual, +8 CTexture*, +0x12 trozo deseado, +0x13 maximo, +0x14 minimo,
//    +0x18 = (tex+0x280)<<24 | uso, +0x1c = texeles residentes de MAS (mip actual vs mip que pide la GPU;
//    0x80000000|.. si no se muestreo), +0x20 = fotogramas con el mismo deseado.
//  - 0xac5b70 calcula el deseado (+0x12) desde el mip que de verdad pide el fotograma (lo cercano pide mip 0-1,
//    lo lejano/pequeno en pantalla pide mips gruesos). tex+0x6c>0 (fijada) => deseado = maximo.
//  - Recorte en 0xac9d80 cuando no cabe:
//    pasada 1 (0xacafef): solo peticiones con actual > deseado, actual > minimo y +0x20 > "Min Num Frames Before
//      Unload" (*0x2f69c98); orden 0xacc070 modo 3 (comparador 0xad2f00: no muestreadas primero, luego mas texeles
//      de mas, luego tex+0x6c, luego +0x20). Baja SOLO hasta el deseado del juego: lo cercano conserva su mip.
//    pasada 2 (0xacb265/0xacb432): si aun falta, CUALQUIER textura con tex+0x6c==0 pasa a deseado 0 (trozo minimo).
//      Esto es lo que emborrono lo cercano con el pool al 50 %.
//  - "Cache Unused Textures" (*0x2f695b8, leido en 0xac9f05): si es 0, la pasada 1 se hace SIEMPRE y entera (suelta
//    todos los mips de mas que el juego ya no pide), sin depender del pool; si es 1 solo cuando falta pool.
// KYTY_VRAM_LEJANAS=1 (por defecto): con la VRAM por encima del presupuesto se pone "Cache Unused Textures"=0; se
// vuelve a 1 cuando sobra VRAM. El pool NO se toca, asi la pasada 2 (la que emborrona lo cercano) no se provoca.
//
// Aqui (solo con KYTY_VRAM_JUEGO=1, apagado por defecto): cada ~2 s, si la VRAM pasa del presupuesto, se baja el "Streaming Pool Total Size MiBs" del juego lo que sobra
// (+ margen); el juego suelta mips (op 2) por su orden de prioridad. Si sobra VRAM, se devuelve poco a poco hacia el
// valor original. Nunca por debajo de KYTY_VRAM_JUEGO_MIN (porcentaje del original, por defecto 50).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32) && (defined(__clang__) || defined(__GNUC__))
extern "C" __declspec(dllimport) int __stdcall IsBadReadPtr(const void*, unsigned long long);
extern "C" __declspec(dllimport) int __stdcall IsBadWritePtr(void*, unsigned long long);
#define KYTY_VRAMJUEGO_OK 1
#else
#define KYTY_VRAMJUEGO_OK 0
#endif

namespace VramJuego {

constexpr uint64_t kEbootBase   = 0x900000000ull;
constexpr uint64_t kPoolPtrRva  = 0x2f69898; // puntero al int "Streaming Pool Total Size MiBs"
constexpr uint64_t kReaderRva   = 0xac9f50;  // mov rax, [rip+disp] dentro del gestor 0xac9ea0
constexpr int64_t  kMarginMiB   = 256;       // se baja lo que sobra + esto (para no rozar el limite)
constexpr int64_t  kSlackMiB    = 768;       // hueco de VRAM a partir del cual se devuelve pool
constexpr int64_t  kStepUpMiB   = 128;       // subida por pasada (lenta: evita el ir y venir)

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_JUEGO");
		return v != nullptr && v[0] == '1'; // apagado: el usuario no quiere que se toque la calidad de las texturas
	}();
	return on;
}

struct State {
	bool     checked = false;
	bool     valid   = false;
	int*     value   = nullptr;
	int      orig    = 0;
	int      floor   = 0;
	int      written = 0;  // ultimo valor puesto por nosotros (0 = ninguno)
	uint64_t ticks   = 0;
};
inline State& S() {
	static State s;
	return s;
}

// Comprueba que es el eboot 01.007.000: la instruccion de 0xac9f50 debe leer justo 0x2f69898.
inline bool CheckEboot() {
#if KYTY_VRAMJUEGO_OK
	const auto* at = reinterpret_cast<const uint8_t*>(kEbootBase + kReaderRva);
	if (IsBadReadPtr(at, 7) != 0) return false;
	if (at[0] != 0x48 || at[1] != 0x8b || at[2] != 0x05) return false;
	int32_t disp = 0;
	std::memcpy(&disp, at + 3, 4);
	return kReaderRva + 7 + static_cast<int64_t>(disp) == kPoolPtrRva;
#else
	return false;
#endif
}

constexpr uint64_t kCachePtrRva  = 0x2f695b8; // puntero al valor "Cache Unused Textures"
constexpr uint64_t kCacheRdRva   = 0xac9f05;  // mov rax, [rip->0x2f695b8] en 0xac9d80

inline bool LejanasOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_LEJANAS");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

struct Lej {
	bool          checked = false;
	bool          valid   = false;
	volatile uint8_t* cache = nullptr;
	uint8_t       orig    = 0;
	bool          off     = false; // lo hemos puesto a 0
	uint64_t      ticks   = 0;
};
inline Lej& L() {
	static Lej l;
	return l;
}

inline bool CheckRip(uint64_t rva, uint64_t target) {
#if KYTY_VRAMJUEGO_OK
	const auto* at = reinterpret_cast<const uint8_t*>(kEbootBase + rva);
	if (IsBadReadPtr(at, 7) != 0) return false;
	if (at[0] != 0x48 || at[1] != 0x8b || at[2] != 0x05) return false;
	int32_t disp = 0;
	std::memcpy(&disp, at + 3, 4);
	return rva + 7 + static_cast<int64_t>(disp) == target;
#else
	(void)rva;
	(void)target;
	return false;
#endif
}

// Solo lejanas: el juego suelta los mips que el mismo ya no pide (pasada 1). Nunca baja de su deseado.
inline void TickLejanas(uint64_t used, uint64_t budget) {
#if KYTY_VRAMJUEGO_OK
	if (!LejanasOn() || budget == 0) return;
	auto& l = L();
	if (!l.checked) {
		l.checked = true;
		l.valid   = CheckRip(kCacheRdRva, kCachePtrRva);
		if (!l.valid) {
			std::printf("[vram-lejanas] eboot no es 01.007.000 (0xac9f05 no lee 0x2f695b8): apagado\n");
			std::fflush(stdout);
			return;
		}
	}
	if (!l.valid) return;
	if (l.cache == nullptr) {
		auto* const* pp = reinterpret_cast<uint8_t* const*>(kEbootBase + kCachePtrRva);
		if (IsBadReadPtr(pp, 8) != 0) return;
		uint8_t* p = *pp;
		if (p == nullptr || IsBadReadPtr(p, 1) != 0 || IsBadWritePtr(p, 1) != 0) return; // opcion aun sin crear
		l.cache = p;
		l.orig  = *l.cache;
		std::printf("[vram-lejanas] KYTY_VRAM_LEJANAS=1: Cache Unused Textures (*0x2f695b8) = %u; pool sin tocar\n",
		            static_cast<unsigned>(l.orig));
		std::fflush(stdout);
	}
	if (l.orig == 0) return; // el juego ya suelta siempre: nada que hacer
	const int64_t used_mib   = static_cast<int64_t>(used >> 20u);
	const int64_t budget_mib = static_cast<int64_t>(budget >> 20u);
	l.ticks++;
	if (!l.off && used_mib > budget_mib) {
		*l.cache = 0;
		l.off    = true;
		std::printf("[vram-lejanas] VRAM %lld > presupuesto %lld MiB: Cache Unused Textures 1 -> 0 (el juego suelta "
		            "los mips que ya no pide; lo cercano conserva su mip)\n",
		            static_cast<long long>(used_mib), static_cast<long long>(budget_mib));
		std::fflush(stdout);
	} else if (l.off && used_mib + kSlackMiB < budget_mib) {
		*l.cache = l.orig;
		l.off    = false;
		std::printf("[vram-lejanas] VRAM %lld, sobran >%lld MiB: Cache Unused Textures 0 -> %u\n",
		            static_cast<long long>(used_mib), static_cast<long long>(kSlackMiB), static_cast<unsigned>(l.orig));
		std::fflush(stdout);
	} else if (l.ticks % 15 == 0) {
		std::printf("[vram-lejanas] 30 s: cache %s; VRAM %lld / presupuesto %lld MiB\n", l.off ? "0 (soltando)" : "1",
		            static_cast<long long>(used_mib), static_cast<long long>(budget_mib));
		std::fflush(stdout);
	}
#else
	(void)used;
	(void)budget;
#endif
}

// (VRAM-TECHO, 11-25) KYTY_VRAM_TECHO=1 (por defecto; =0 lo apaga, y entonces manda KYTY_VRAM_LEJANAS como antes).
// Log 20261125-050455: [vram] used 5863 / budget 5449 MiB, 11796 expulsiones (38093 MiB) -> texturas raras y sombras
// cortadas. Lejanas solo actuaba POR ENCIMA del presupuesto (tarde) y con "Min Num Frames Before Unload" de fabrica.
// Ghidra 01.007.000:
//  - "Min Num Frames Before Unload": objeto 0x2f69c38 (registro 0xac733c en 0xac6310), su valor es un int en
//    *[0x2f69c98] (0xad4f0b pone el puntero a 0x2f69d30; minimo 0, maximo 0x7fffffff).
//  - Sus 4 lectores (0xac62e4, 0xacabc8 = pasada 1 de 0xac9d80, 0xacbe9c, 0xad3070) hacen la MISMA prueba:
//    actual(+0) > deseado(+0x12) && actual > minimo(+0x14) && fotogramas con el mismo deseado(+0x20) > MinFrames.
//    Es decir: solo toca texturas que tienen MAS mip del que la GPU pide (lejanas/pequenas en pantalla). Una textura
//    cercana (deseado = actual) nunca entra. La pasada 2 (la que emborrona lo cercano) solo la provoca bajar el pool,
//    que aqui NO se toca.
// Aqui: con la VRAM por encima del 90 % del presupuesto (antes de llegar), "Cache Unused Textures"=0 (pasada 1 siempre)
// y MinFrames = KYTY_VRAM_TECHO_FRAMES (por defecto 2): el juego baja ya la resolucion de las lejanas (op 2 -> mip
// desligado sparse, VRAM devuelta). Bajo el 80 % se devuelven los dos valores originales.
inline bool TechoOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_VRAM_TECHO");
		return v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	}();
	return on;
}

constexpr uint64_t kMinFramesPtrRva = 0x2f69c98; // puntero al int "Min Num Frames Before Unload"
constexpr uint64_t kMinFramesRdRva  = 0xacabc8;  // mov rcx, [rip->0x2f69c98] (pasada 1 de 0xac9d80)

struct Techo {
	bool              checked = false, valid = false, activo = false;
	volatile uint8_t* cache   = nullptr;
	volatile int*     frames  = nullptr;
	uint8_t           cache_orig  = 0;
	int               frames_orig = 0;
	uint64_t          last_log_ms = 0, veces = 0;
	uint64_t          mips0 = 0, freed0 = 0;           // contadores al empezar la ventana de 5 s
	uint64_t          mips_total = 0, freed_total = 0; // solo mientras el techo esta activo
};
inline Techo& T() {
	static Techo t;
	return t;
}

inline bool CheckRipRcx(uint64_t rva, uint64_t target) {
#if KYTY_VRAMJUEGO_OK
	const auto* at = reinterpret_cast<const uint8_t*>(kEbootBase + rva);
	if (IsBadReadPtr(at, 7) != 0) return false;
	if (at[0] != 0x48 || at[1] != 0x8b || at[2] != 0x0d) return false;
	int32_t disp = 0;
	std::memcpy(&disp, at + 3, 4);
	return rva + 7 + static_cast<int64_t>(disp) == target;
#else
	(void)rva;
	(void)target;
	return false;
#endif
}

// Cada ~1 s desde el hilo de render. used/budget en bytes; now_ms reloj monotono; mips_unbound / freed_bytes:
// contadores acumulados de mips nativos desligados y bytes de VRAM devueltos (TexNativa); native_bound: VRAM ligada ahora.
inline void TickTecho(uint64_t used, uint64_t budget, uint64_t now_ms, uint64_t mips_unbound, uint64_t freed_bytes,
                      uint64_t native_bound) {
#if KYTY_VRAMJUEGO_OK
	if (!TechoOn() || budget == 0) return;
	auto& t = T();
	if (!t.checked) {
		t.checked = true;
		t.valid   = CheckRip(kCacheRdRva, kCachePtrRva) && CheckRipRcx(kMinFramesRdRva, kMinFramesPtrRva);
		t.mips0   = mips_unbound;
		t.freed0  = freed_bytes;
		t.last_log_ms = now_ms;
		if (!t.valid) {
			std::printf("[vram-techo] eboot no es 01.007.000 (0xac9f05/0xacabc8 no leen 0x2f695b8/0x2f69c98): apagado\n");
			std::fflush(stdout);
			return;
		}
	}
	if (!t.valid) return;
	if (t.cache == nullptr || t.frames == nullptr) {
		auto* const* pc = reinterpret_cast<uint8_t* const*>(kEbootBase + kCachePtrRva);
		auto* const* pf = reinterpret_cast<int* const*>(kEbootBase + kMinFramesPtrRva);
		if (IsBadReadPtr(pc, 8) != 0 || IsBadReadPtr(pf, 8) != 0) return;
		uint8_t* c = *pc;
		int*     f = *pf;
		if (c == nullptr || f == nullptr || IsBadWritePtr(c, 1) != 0 || IsBadWritePtr(f, 4) != 0) return; // aun sin crear
		const int fv = *reinterpret_cast<volatile int*>(f);
		if (fv < 0 || fv > 100000) return; // aun sin valor real
		t.cache       = c;
		t.frames      = f;
		t.cache_orig  = *t.cache;
		t.frames_orig = fv;
		std::printf("[vram-techo] KYTY_VRAM_TECHO=1: techo 90 %% del presupuesto; Cache Unused Textures (*0x2f695b8) = %u, "
		            "Min Num Frames Before Unload (*0x2f69c98) = %d; pool del juego sin tocar (lo cercano no se toca)\n",
		            static_cast<unsigned>(t.cache_orig), t.frames_orig);
		std::fflush(stdout);
	}
	static const int frames_techo = [] {
		// (VG-AB, 11-25) Por defecto el valor del propio juego (-1): con 2, log 102943 solto 108-232 MiB de mips cada 5 s
		// y re-subio 86-212 MiB en las mismas ventanas (ir y venir); el uso no bajo del tope en todo el minuto.
		int v = -1;
		if (const char* e = std::getenv("KYTY_VRAM_TECHO_FRAMES")) v = std::atoi(e);
		return v;
	}();
	const uint64_t used_mib   = used >> 20u;
	const uint64_t budget_mib = budget >> 20u;
	const uint64_t hi_mib     = budget_mib * 90 / 100;
	const uint64_t lo_mib     = budget_mib * 80 / 100;
	if (!t.activo && used_mib > hi_mib) {
		*t.cache  = 0;
		*t.frames = frames_techo >= 0 && frames_techo < t.frames_orig ? frames_techo : t.frames_orig;
		t.activo  = true;
		t.veces++;
		std::printf("[vram-techo] VRAM %llu > techo %llu MiB (90 %% de %llu): el juego baja la resolucion de las texturas "
		            "lejanas (Cache Unused 0, Min Frames %d -> %d)\n",
		            static_cast<unsigned long long>(used_mib), static_cast<unsigned long long>(hi_mib),
		            static_cast<unsigned long long>(budget_mib), t.frames_orig, *t.frames);
		std::fflush(stdout);
	} else if (t.activo && used_mib < lo_mib) {
		*t.cache  = t.cache_orig;
		*t.frames = t.frames_orig;
		t.activo  = false;
		std::printf("[vram-techo] VRAM %llu < %llu MiB (80 %%): valores del juego devueltos (Cache Unused %u, Min Frames %d)\n",
		            static_cast<unsigned long long>(used_mib), static_cast<unsigned long long>(lo_mib),
		            static_cast<unsigned>(t.cache_orig), t.frames_orig);
		std::fflush(stdout);
	} else if (t.activo) {
		// el juego (o su config) puede haber reescrito los valores: se mantienen mientras dure el techo
		if (*t.cache != 0) *t.cache = 0;
		const int want = frames_techo < t.frames_orig ? frames_techo : t.frames_orig;
		if (*t.frames != want) *t.frames = want;
	}
	if (now_ms - t.last_log_ms >= 5000) {
		const uint64_t mips  = mips_unbound - t.mips0;
		const uint64_t freed = freed_bytes - t.freed0;
		if (t.activo) {
			t.mips_total += mips;
			t.freed_total += freed;
		}
		std::printf("[vram-techo] 5 s: used %llu / budget %llu MiB (%llu %%; techo %llu); %s; texturas bajadas de resolucion "
		            "(mips lejanos soltados) %llu, MiB liberados %llu (con techo, total: %llu mips, %llu MiB; activado %llu "
		            "veces); nativas ligadas %llu MiB\n",
		            static_cast<unsigned long long>(used_mib), static_cast<unsigned long long>(budget_mib),
		            static_cast<unsigned long long>(budget_mib ? used_mib * 100 / budget_mib : 0),
		            static_cast<unsigned long long>(hi_mib), t.activo ? "ACTIVO" : "en reposo",
		            static_cast<unsigned long long>(mips), static_cast<unsigned long long>(freed >> 20u),
		            static_cast<unsigned long long>(t.mips_total), static_cast<unsigned long long>(t.freed_total >> 20u),
		            static_cast<unsigned long long>(t.veces), static_cast<unsigned long long>(native_bound >> 20u));
		std::fflush(stdout);
		t.last_log_ms = now_ms;
		t.mips0       = mips_unbound;
		t.freed0      = freed_bytes;
	}
#else
	(void)used;
	(void)budget;
	(void)now_ms;
	(void)mips_unbound;
	(void)freed_bytes;
	(void)native_bound;
#endif
}

// used/budget: VRAM usada y presupuesto (bytes) de la cache de texturas. Llamar cada ~2 s desde el hilo de render.
inline void Tick(uint64_t used, uint64_t budget) {
	if (!TechoOn()) TickLejanas(used, budget); // (VRAM-TECHO) el techo lleva Cache Unused y Min Frames
#if KYTY_VRAMJUEGO_OK
	if (!On() || budget == 0) return;
	auto& s = S();
	if (!s.checked) {
		s.checked = true;
		s.valid   = CheckEboot();
		if (!s.valid) {
			std::printf("[vram-juego] eboot no es 01.007.000 (0xac9f50 no lee 0x2f69898): apagado\n");
			std::fflush(stdout);
			return;
		}
	}
	if (!s.valid) return;
	if (s.value == nullptr) {
		auto* const* pp = reinterpret_cast<int* const*>(kEbootBase + kPoolPtrRva);
		if (IsBadReadPtr(pp, 8) != 0) return;
		int* p = *pp;
		if (p == nullptr || IsBadReadPtr(p, 4) != 0 || IsBadWritePtr(p, 4) != 0) return; // aun sin crear la opcion
		const int v = *reinterpret_cast<volatile int*>(p);
		if (v < 64 || v > 65536) return; // aun sin valor real
		s.value = p;
		s.orig  = v;
		int pct = 85; // (log 060620: al 50 % las texturas cercanas se veian borrosas y la VRAM apenas bajaba)
		if (const char* e = std::getenv("KYTY_VRAM_JUEGO_MIN")) pct = std::atoi(e);
		if (pct < 10) pct = 10;
		if (pct > 100) pct = 100;
		s.floor = static_cast<int>(static_cast<int64_t>(v) * pct / 100);
		std::printf("[vram-juego] KYTY_VRAM_JUEGO=1: pool de streaming del juego (Streaming Pool Total Size, *0x2f69898) "
		            "original %d MiB, minimo %d MiB (%d %%); VRAM %llu / presupuesto %llu MiB\n",
		            v, s.floor, pct, static_cast<unsigned long long>(used >> 20u),
		            static_cast<unsigned long long>(budget >> 20u));
		std::fflush(stdout);
	}
	volatile int* vp  = s.value;
	int           cur = *vp;
	if (s.written != 0 && cur != s.written) { // el juego (o su config) lo cambio: ese es el nuevo original
		std::printf("[vram-juego] el juego cambio el pool a %d MiB (puesto %d): nuevo original\n", cur, s.written);
		s.orig    = cur;
		s.floor   = cur / 2 > s.floor ? cur / 2 : s.floor;
		s.written = 0;
	}
	const int64_t used_mib   = static_cast<int64_t>(used >> 20u);
	const int64_t budget_mib = static_cast<int64_t>(budget >> 20u);
	int           next       = cur;
	if (used_mib > budget_mib) {
		next = static_cast<int>(static_cast<int64_t>(cur) - (used_mib - budget_mib) - kMarginMiB);
		if (next < s.floor) next = s.floor;
	} else if (used_mib + kSlackMiB < budget_mib && cur < s.orig) {
		next = cur + static_cast<int>(kStepUpMiB);
		if (next > s.orig) next = s.orig;
	}
	s.ticks++;
	if (next != cur) {
		*vp       = next;
		s.written = next;
		std::printf("[vram-juego] pool de streaming del juego %d -> %d MiB (original %d, minimo %d); VRAM %lld / "
		            "presupuesto %lld MiB\n",
		            cur, next, s.orig, s.floor, static_cast<long long>(used_mib), static_cast<long long>(budget_mib));
		std::fflush(stdout);
	} else if (s.ticks % 15 == 0) { // ~30 s
		std::printf("[vram-juego] 30 s: pool %d MiB (original %d, minimo %d); VRAM %lld / presupuesto %lld MiB\n", cur,
		            s.orig, s.floor, static_cast<long long>(used_mib), static_cast<long long>(budget_mib));
		std::fflush(stdout);
	}
#else
	(void)used;
	(void)budget;
#endif
}

} // namespace VramJuego
