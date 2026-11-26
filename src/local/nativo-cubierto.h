#pragma once
// (VIGILANCIA-DOBLE) KYTY_SIN_VIGILANCIA_DOBLE (por defecto 1; =0 todo como antes).
//
// Log 20261125-061906-403 ([raiz] + [frame-nativo]): 52 816 cambios de proteccion/s del emulador. Por zona (30 s):
// DefaultTagg 23 878/s (42 %, 613 638 pag/s), otros 20 734/s (36 %), cpuheap 11 969/s (21 %), cpubkt 291/s, gpupage 19/s.
// Por motivo (PROTECT BY ORIGIN): hilo GPU ~20 100/s a 60.6 fps (read-flush 142/frame = 8 617/s, write-release 172/frame = 10 441/s, write-track 18/frame = 1 073/s),
// otros hilos reapply 12 944/s (287 693 pag/s), deferred-release 11 325/s, write-release 8 455/s. Es decir, la vigilancia
// del emulador sigue protegiendo paginas que ya sirve otra via (anuncios del juego, piezas nativas).
//
// Este header da dos cosas:
//  1) API para piezas nativas (mallas, texturas, sombras...): "estas paginas las sirvo yo". Una pagina cubierta queda
//     FUERA de la vigilancia del emulador del todo: el kernel (ProtectMappedUnlocked) no le aplica ninguna proteccion de
//     vigilancia (read-track, read-flush, write-track, reapply, sync a solo-lectura/sin acceso). Las que ya tuviera se
//     quitan la proxima vez que el emulador la suelte (write-release / deferred-release siguen pasando: solo abren).
//       NativoCubierto::MarcarCubierto(addr, size, dueno)   // dueno: NativoCubierto::DMallas, DTexturas, DSombras, ...
//       NativoCubierto::Descubrir(addr, size)               // la pieza deja de servirla (vuelve a la vigilancia)
//       NativoCubierto::Cubierto(addr) / RangoCubierto(addr, size)
//     Contrato: la pieza que marca es la UNICA via por la que la GPU lee esas paginas (no la cache de buffers/texturas del
//     emulador). SEGURIDAD: si la cache del emulador llega a pedir un rango cubierto (BufferCache::PolicyBeforeUseSlow),
//     se Descubre (vuelve a la vigilancia normal, CPU-sucio = se resube) y se cuenta en la linea [vigilancia].
//     Granularidad: pagina de 4 KiB (Marcar y Descubrir hacia dentro: solo paginas enteras; dueno por pagina).
//  2) Contadores de protecciones por zona x motivo y la linea "[vigilancia]" cada 5 s (antes/ahora).
#include "frame-nativo.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace NativoCubierto {

[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		// (Apagado por defecto) Log 20261125-065754-173 vs 061906-403 (mismo cmd 33): faults de escritura del juego 582/s ->
		// 4405/s; el anuncio "frame = escrita por la GPU 0 -> 1380, VETADA antes 0 -> 824" y la VERIFY "SIN ANUNCIO" 13 739
		// paginas/5 s (vetadas -> fault -> reentregadas -> otra vez), mas 1,5 M ResetWriteWatch/5 s; 60 -> 34-45 fps. Solo
		// con KYTY_SIN_VIGILANCIA_DOBLE=1.
		const char* v = std::getenv("KYTY_SIN_VIGILANCIA_DOBLE");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

enum Dueno : uint8_t { DNadie = 0, DMallas, DTexturas, DSombras, DFrame, DOtro, DCount };
inline const char* const kDuenoNames[DCount] = {"-", "mallas", "texturas", "sombras", "frame", "otro"};

constexpr uint64_t kLimit = uint64_t {1} << 37u;         // 128 GiB de VA del juego
constexpr uint64_t kPages = kLimit >> 12u;               // 32 Mi paginas
constexpr uint64_t kWords = kPages / 64u;                // 4 MiB de bits

inline std::atomic<std::atomic<uint64_t>*> g_bits {nullptr};
inline std::atomic<int64_t>                g_pages[DCount] {};  // paginas cubiertas ahora, por dueno
inline std::atomic<uint64_t>               g_skipped_calls {0}, g_skipped_pages {0}; // protecciones no hechas
inline std::atomic<uint64_t>               g_partial {0};       // protecciones con parte cubierta (hechas: ver abajo)
inline std::atomic<uint64_t>               g_safety {0}, g_safety_pages {0}; // la cache del emulador pidio un rango cubierto

[[nodiscard]] inline std::atomic<uint64_t>* Bits(bool create) noexcept {
	auto* b = g_bits.load(std::memory_order_acquire);
	if (b != nullptr || !create) return b;
	auto* n = new (std::nothrow) std::atomic<uint64_t>[kWords];
	if (n == nullptr) return nullptr;
	for (uint64_t i = 0; i < kWords; ++i) n[i].store(0, std::memory_order_relaxed);
	std::atomic<uint64_t>* expected = nullptr;
	if (!g_bits.compare_exchange_strong(expected, n, std::memory_order_acq_rel)) {
		delete[] n;
		return expected;
	}
	return n;
}

// Dueno por pagina (revision 11-25): antes el mapa de bits no guardaba dueno y Descubrir redondeaba hacia fuera, asi
// que descubrir un dueno destapaba paginas de otro y g_pages no cuadraba. Ahora cada pagina tiene un dueno (tabla
// dispersa: trozos de 64 Ki paginas = 64 KiB, creados al marcar); una pagina ya de otro dueno no se roba, y Descubrir
// con dueno solo quita las suyas. Marcar y Descubrir redondean los dos HACIA DENTRO (solo paginas enteras del rango).
constexpr uint64_t kChunkShift = 16u;
constexpr uint64_t kChunkMask  = (uint64_t {1} << kChunkShift) - 1u;
constexpr uint64_t kChunks     = kPages >> kChunkShift; // 512
inline std::atomic<std::atomic<uint8_t>*> g_owner[kChunks] {};

[[nodiscard]] inline std::atomic<uint8_t>* OwnerChunk(uint64_t page, bool create) noexcept {
	auto& slot = g_owner[page >> kChunkShift];
	auto* c    = slot.load(std::memory_order_acquire);
	if (c != nullptr || !create) return c;
	auto* n = new (std::nothrow) std::atomic<uint8_t>[kChunkMask + 1u];
	if (n == nullptr) return nullptr;
	for (uint64_t i = 0; i <= kChunkMask; ++i) n[i].store(0, std::memory_order_relaxed);
	std::atomic<uint8_t>* expected = nullptr;
	if (!slot.compare_exchange_strong(expected, n, std::memory_order_acq_rel)) {
		delete[] n;
		return expected;
	}
	return n;
}

// [addr, addr + size). cover: paginas enteras, sin robar las de otro dueno. !cover: dueno != DNadie -> solo las suyas;
// dueno == DNadie -> las de cualquiera, restando al dueno real. outward (solo SEGURIDAD): toda pagina que toque el rango.
// Devuelve las paginas que cambiaron de estado.
inline uint64_t SetRange(uint64_t addr, uint64_t size, bool cover, uint8_t dueno, bool outward = false) noexcept {
	if (!On() || size == 0 || addr >= kLimit || dueno >= DCount || (cover && dueno == DNadie)) return 0;
	auto* b = Bits(cover);
	if (b == nullptr) return 0;
	const uint64_t end   = addr + size < kLimit ? addr + size : kLimit;
	uint64_t       first = (addr + 0xfffu) >> 12u, last = end >> 12u; // [first, last), hacia dentro
	if (outward) {
		first = addr >> 12u;
		last  = (end + 0xfffu) >> 12u;
		if (last > kPages) last = kPages;
	}
	uint64_t changed       = 0;
	int64_t  delta[DCount] = {};
	for (uint64_t p = first; p < last; ++p) {
		auto* c = OwnerChunk(p, cover);
		if (c == nullptr) { // (trozo sin crear: nada cubierto en el)
			p |= kChunkMask;
			continue;
		}
		auto&          o   = c[p & kChunkMask];
		const uint64_t bit = uint64_t {1} << (p & 63u);
		if (cover) {
			uint8_t exp = DNadie;
			if (!o.compare_exchange_strong(exp, dueno, std::memory_order_acq_rel)) continue; // (suya ya, o de otro)
			b[p >> 6u].fetch_or(bit, std::memory_order_acq_rel);
			++delta[dueno];
			++changed;
		} else {
			uint8_t cur = o.load(std::memory_order_acquire);
			if (cur == DNadie || (dueno != DNadie && cur != dueno)) continue;
			b[p >> 6u].fetch_and(~bit, std::memory_order_acq_rel);
			if (!o.compare_exchange_strong(cur, DNadie, std::memory_order_acq_rel)) continue;
			--delta[cur];
			++changed;
		}
	}
	for (int d = 1; d < DCount; ++d)
		if (delta[d] != 0) g_pages[d].fetch_add(delta[d], std::memory_order_relaxed);
	return changed;
}

inline uint64_t MarcarCubierto(uint64_t addr, uint64_t size, uint8_t dueno) noexcept { return SetRange(addr, size, true, dueno); }
// Con dueno: solo sus paginas enteras. DNadie: las enteras del rango, de quien sean (el contador baja al dueno real).
inline uint64_t Descubrir(uint64_t addr, uint64_t size, uint8_t dueno = DNadie) noexcept { return SetRange(addr, size, false, dueno); }

[[nodiscard]] inline bool Cubierto(uint64_t addr) noexcept {
	auto* b = g_bits.load(std::memory_order_acquire);
	if (b == nullptr || addr >= kLimit) return false;
	const uint64_t p = addr >> 12u;
	return (b[p >> 6u].load(std::memory_order_relaxed) >> (p & 63u) & 1u) != 0;
}
// 0 nada cubierto, 1 parte, 2 todo.
[[nodiscard]] inline int Cobertura(uint64_t addr, uint64_t size) noexcept {
	auto* b = g_bits.load(std::memory_order_acquire);
	if (b == nullptr || size == 0 || addr >= kLimit) return 0;
	const uint64_t first = addr >> 12u, last = ((addr + size - 1) >> 12u);
	uint64_t       in    = 0;
	for (uint64_t p = first; p <= last; ++p)
		if (p < kPages && (b[p >> 6u].load(std::memory_order_relaxed) >> (p & 63u) & 1u) != 0) ++in;
	return in == 0 ? 0 : (in == last - first + 1 ? 2 : 1);
}
[[nodiscard]] inline bool RangoCubierto(uint64_t addr, uint64_t size) noexcept { return Cobertura(addr, size) == 2; }

// ---- Kernel (ProtectMappedUnlocked): una proteccion de VIGILANCIA sobre paginas todas cubiertas no se hace. ----------
// origin = Raiz::t_protect_origin (1 read-track, 2 read-flush, 4 write-track, 7 sync, 8 reapply). `protecting`: el modo
// pedido es solo-lectura o sin acceso. Las que abren (release) siempre pasan: dejan la pagina RW, que es lo que queremos.
[[nodiscard]] inline bool SkipWatch(uint64_t vaddr, uint64_t size, uint32_t origin, bool protecting) noexcept {
	if (!protecting || g_bits.load(std::memory_order_relaxed) == nullptr || !On()) return false;
	if (origin != 1 && origin != 2 && origin != 4 && origin != 7 && origin != 8) return false;
	const int c = Cobertura(vaddr, size);
	if (c == 2) {
		g_skipped_calls.fetch_add(1, std::memory_order_relaxed);
		g_skipped_pages.fetch_add((size + 0xfffu) >> 12u, std::memory_order_relaxed);
		return true;
	}
	// Parte cubierta: se hace entera (el gestor de paginas cree que quedo protegida; partirla dejaria su estado mal).
	if (c == 1) g_partial.fetch_add(1, std::memory_order_relaxed);
	return false;
}

// ---- SEGURIDAD: la cache del emulador pide un rango cubierto -> vuelve a la vigilancia (el llamador lo pone CPU-sucio).
inline bool SeguridadUsoEmulador(uint64_t vaddr, uint64_t size) noexcept {
	if (g_bits.load(std::memory_order_relaxed) == nullptr || Cobertura(vaddr, size) == 0) return false;
	g_safety.fetch_add(1, std::memory_order_relaxed);
	g_safety_pages.fetch_add(SetRange(vaddr, size, false, DNadie, true), std::memory_order_relaxed); // (toda pagina tocada)
	return true;
}

// ---- Contadores por zona (FrameNativo::Classify) x motivo (Raiz origin) -------------------------------------------
constexpr uint32_t kOrigins = 11; // Raiz::ProtOriginCount
inline const char* const kOriginNames[kOrigins] = {"other", "read-track", "read-flush", "read-release", "write-track", "write-release",
                                                   "offload", "sync", "reapply", "restore", "deferred-release"};
inline std::atomic<uint64_t> g_zo[FrameNativo::CCount][kOrigins] {};
inline void NoteProtect(uint64_t vaddr, uint32_t origin) noexcept {
	g_zo[FrameNativo::Classify(vaddr)][origin < kOrigins ? origin : 0].fetch_add(1, std::memory_order_relaxed);
}
// Paginas de frame que la politica de frame (anuncios del juego: ganchos 0x811f60 / 0xa08ae0 / 0xa188bc) deja sin vigilancia.
inline std::atomic<uint64_t> g_frame_nowatch {0};
inline void NoteFrameNoWatch() noexcept { g_frame_nowatch.fetch_add(1, std::memory_order_relaxed); }

} // namespace NativoCubierto

// ---- (CPUHEAP-SIN-VIGILAR) KYTY_CPUHEAP_SIN_VIGILAR (por defecto 1; =0 todo como antes) ------------------------------
// Log 20261125-072323-031: cpuheap ~12 000 protecciones/s (21 %) con solo ~16 900 pag/s (1,35 paginas por llamada).
// Ghidra (eboot 01.007.000): CBuffer::Create 0x909460 reserva los datos con FUN_007f9cf0(bytes, categoria, alin):
// flag 1 -> 0x2ea0d00 (GPUMemory, gpuheap), sin flag 1 / 2 / 0x800 / 0x1000 -> 0x2ea0d18 (CPUMemory = cpuheap, la misma
// categoria de los vectores del motor). Los CBuffer con flag 4 / 0x14 y sin datos iniciales son buffers que ESCRIBE la GPU
// (VirtualGeometryRendering::VirtualGeometryRendering 0xa53fa0: contadores de 4 B x 0x81/0x41/0x200; ParticleRenderBuffers
// 0xa913d0: 0x44 x 0x10000, 0x78 x 0x20000...; FurRenderEntity::FinalizeCreation...), 105 llamadas con flag 4: viven en
// cpuheap, alineados a 16-32 B, en las mismas paginas de 4 KiB que objetos que la CPU escribe cada frame y las cabeceras
// del HeapAllocator (escrituras de eboot+0x802a6c..0x8035cf en [anuncio]/[write-policy] "escrita por la GPU (cpuheap)").
// Cada escritura de la GPU ponia esas paginas SIN ACCESO (vigilante de lectura: read-track/read-flush) y la devolucion
// las volvia a abrir (deferred-release / read-release): ~2 cambios por pagina y frame, y la lectura/escritura de la CPU
// de un objeto vecino caia en fault.
// Ahora: una pagina de cpuheap que la GPU toma NO recibe vigilante de lectura (RegionManager::UpdateGpuProtection):
// conserva su proteccion de escritura (solo-lectura si estaba limpia: una escritura de la CPU sigue haciendo fault y el
// fault lee de vuelta lo de la GPU antes, InvalidateRegion -> ReadMemory, como antes), y lo que escribe la GPU llega a la
// memoria del juego por las devoluciones de siempre (readback / feedback / download: WriteBacking) y por el gancho de
// lectura de CBuffer (0x909940/0x909970). Nada se hace dos veces: no hay copia aparte ni vigilancia nueva.
namespace CpuheapSinVigilar {
[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_CPUHEAP_SIN_VIGILAR");
		const bool  yes = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
		std::printf("[vigilancia] KYTY_CPUHEAP_SIN_VIGILAR=%d: %s\n", yes ? 1 : 0,
		            yes ? "ENCENDIDO por defecto: las paginas de cpuheap que escribe la GPU no se ponen sin acceso "
		                  "(sin read-track / deferred-release); =0 lo apaga"
		                : "apagado (como antes)");
		std::fflush(stdout);
		return yes;
	}();
	return on;
}
inline std::atomic<uint64_t> g_taken {0};    // paginas de cpuheap que la GPU tomo sin ponerlas sin acceso
inline std::atomic<uint64_t> g_given {0};    // y devueltas sin abrirlas (no se habian cerrado)
inline std::atomic<int64_t>  g_now {0};      // paginas asi ahora
} // namespace CpuheapSinVigilar

// ---- (DEFAULTTAGG-SIN-VIGILAR) KYTY_DEFAULTTAGG_SIN_VIGILAR (por defecto 0; =1 lo enciende) ----------------------------
// Igual que el cpuheap, para DefaultTagg (TaggedFrameAllocator del juego; ~24 600 protecciones/s en el log 20261126-022228).
// Ghidra (eboot 01.007.000), lectores de DefaultTagg de la tabla [read-audit] del log 20261126-020330:
//  B (datos de la CPU en la misma pagina): 0x8120b8 (lista libre de TaggedFrameAllocator 0x811f60), 0xab610c/0xab611d/
//    0xab6121/0xab6d83 (contexto de dibujo, 0xab6080/0xab6d00), 0xa0221a (campo de objeto, 0xa00eb0).
//  0x8e9760 (copia vec3, sin gancho por defecto) por llamador: 0x8f6280, 0x9e14d0, 0x9cb610 (registros en pila), 0xa71230
//    (mapa ordenado 0xab6ef0) = B; 0x9a4db0 lee un std::vector de 0x1010 dwords (+0x408/+0x430, creado en 0x9a22c0 con
//    FUN_009a5880) que RELLENA LA CPU con FUN_00909970 (copia de CBuffer, enganchada) en 0x9a4330 = B; 0xa2f910 (rayos de
//    colision GPU) lee registros devueltos por FUN_00909940 (enganchado) = A cubierto.
//  A con gancho: 0xadabb0 (KYTY_DS_READ_HOOK2), 0x909940/0x909970 (KYTY_DS_READBACK_HOOK).
// Mismo mecanismo que CpuheapSinVigilar (RegionManager::UpdateGpuProtection, marca m_noread): sin vigilante de lectura.
namespace DefaultTaggSinVigilar {
[[nodiscard]] inline bool On() noexcept {
	static const bool on = [] {
		const char* v   = std::getenv("KYTY_DEFAULTTAGG_SIN_VIGILAR");
		const bool  yes = (v != nullptr && v[0] == '1');
		std::printf("[vigilancia] KYTY_DEFAULTTAGG_SIN_VIGILAR=%d: %s\n", yes ? 1 : 0,
		            yes ? "ENCENDIDO: las paginas de DefaultTagg que escribe la GPU no se ponen sin acceso" : "apagado (como antes; =1 lo enciende)");
		std::fflush(stdout);
		return yes;
	}();
	return on;
}
inline std::atomic<uint64_t> g_taken {0}; // paginas de DefaultTagg que la GPU tomo sin ponerlas sin acceso
} // namespace DefaultTaggSinVigilar

namespace NativoCubierto {
// Linea [vigilancia] (la llama Raiz::Report, cada 5 s). Antes = log 20261125-061906-403 (sin este cambio).
inline void Report(double secs) noexcept {
	if (secs <= 0.0) return;
	static uint64_t prev[FrameNativo::CCount][kOrigins] = {};
	static uint64_t prev_s[6]                           = {};
	const double    ps                                  = 1.0 / secs;
	static constexpr double antes_zona[FrameNativo::CCount] = {20734, 23878, 0, 0, 11969, 291, 19};
	double zona[FrameNativo::CCount] = {}, motivo[kOrigins] = {}, total = 0, cpuheap_motivo[kOrigins] = {};
	for (int z = 0; z < FrameNativo::CCount; ++z)
		for (uint32_t o = 0; o < kOrigins; ++o) {
			const uint64_t v = g_zo[z][o].load(std::memory_order_relaxed);
			const double   d = static_cast<double>(v - prev[z][o]) * ps;
			prev[z][o]       = v;
			zona[z] += d;
			motivo[o] += d;
			if (z == FrameNativo::CCpuheap) cpuheap_motivo[o] = d;
			total += d;
		}
	const uint64_t s[6] = {g_skipped_calls.load(), g_skipped_pages.load(), g_partial.load(), g_safety.load(), g_safety_pages.load(),
	                       g_frame_nowatch.load()};
	double ds[6];
	for (int i = 0; i < 6; ++i) {
		ds[i]     = static_cast<double>(s[i] - prev_s[i]) * ps;
		prev_s[i] = s[i];
	}
	std::printf("[vigilancia] %.1f s (KYTY_SIN_VIGILANCIA_DOBLE=%d; =0 lo apaga): protecciones/s antes (log 061906) -> ahora: total "
	            "52816 -> %.0f; por zona:",
	            secs, On() ? 1 : 0, total);
	for (int z = 0; z < FrameNativo::CCount; ++z) std::printf(" %s %.0f -> %.0f", FrameNativo::kNames[z], antes_zona[z], zona[z]);
	std::printf("; por motivo ahora:");
	for (uint32_t o = 0; o < kOrigins; ++o)
		if (motivo[o] >= 0.5) std::printf(" %s %.0f", kOriginNames[o], motivo[o]);
	std::printf(" (antes: reapply 12944, deferred-release 11325, write-release 18896, read-flush 8617, write-track 1073)");
	int64_t cub = 0;
	for (int d = 1; d < DCount; ++d) cub += g_pages[d].load(std::memory_order_relaxed);
	std::printf("; paginas excluidas: cubiertas por piezas nativas %lld (", static_cast<long long>(cub));
	for (int d = 1; d < DCount; ++d) std::printf("%s%s %lld", d > 1 ? ", " : "", kDuenoNames[d], static_cast<long long>(g_pages[d].load()));
	std::printf("), protecciones de vigilancia no hechas sobre ellas %.0f/s (%.0f pag/s), con parte cubierta (hechas) %.0f/s; paginas "
	            "de frame dejadas sin vigilancia por anuncio %.0f/s; SEGURIDAD: rangos cubiertos pedidos por la cache del emulador "
	            "(vuelven a la vigilancia) %.0f/s (%.0f pag/s, total %llu)\n",
	            ds[0], ds[1], ds[2], ds[5], ds[3], ds[4], static_cast<unsigned long long>(s[3]));
	// (CPUHEAP-SIN-VIGILAR) cpuheap por motivo, y lo que el cambio dejo sin proteger. Antes: log 072323 cpuheap ~12 000/s.
	{
		static uint64_t prev_c[2] = {};
		const uint64_t  c[2]      = {CpuheapSinVigilar::g_taken.load(), CpuheapSinVigilar::g_given.load()};
		std::printf("[vigilancia] cpuheap (KYTY_CPUHEAP_SIN_VIGILAR=%d): protecciones/s antes (log 072323) ~12000 -> ahora %.0f; "
		            "por motivo:",
		            CpuheapSinVigilar::On() ? 1 : 0, zona[FrameNativo::CCpuheap]);
		for (uint32_t o = 0; o < kOrigins; ++o)
			if (cpuheap_motivo[o] >= 0.5) std::printf(" %s %.0f", kOriginNames[o], cpuheap_motivo[o]);
		std::printf("; paginas que la GPU escribio sin ponerlas sin acceso %.0f/s (devueltas sin abrir %.0f/s), asi ahora %lld\n",
		            static_cast<double>(c[0] - prev_c[0]) * ps, static_cast<double>(c[1] - prev_c[1]) * ps,
		            static_cast<long long>(CpuheapSinVigilar::g_now.load(std::memory_order_relaxed)));
		prev_c[0] = c[0];
		prev_c[1] = c[1];
	}
	{ // (DEFAULTTAGG-SIN-VIGILAR) Antes: log 20261126-022228 DefaultTagg 24632/s.
		static uint64_t prev_d = 0;
		const uint64_t  d      = DefaultTaggSinVigilar::g_taken.load();
		std::printf("[vigilancia] DefaultTagg (KYTY_DEFAULTTAGG_SIN_VIGILAR=%d): protecciones/s antes (log 022228) 24632 -> ahora %.0f; "
		            "paginas que la GPU escribio sin ponerlas sin acceso %.0f/s\n",
		            DefaultTaggSinVigilar::On() ? 1 : 0, zona[FrameNativo::CDefaultTagg], static_cast<double>(d - prev_d) * ps);
		prev_d = d;
	}
	std::fflush(stdout);
}

} // namespace NativoCubierto
