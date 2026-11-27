#pragma once
// (techo-faltas) Con KYTY_TECHO=1 (solo mide; apagado = una lectura de un static por falta): censo de las faltas de
// pagina de la vigilancia (GpuResourceManager::HandleFault) en los hilos del juego, para saber QUE memoria escribe el
// juego y QUIEN (log 20261127-101545-220, Boletaria 1-1: 7.04 ms/fot y ~236 faltas/fot en hilos del juego, ~30 us cada
// una; la vigilancia de cpuheap/DefaultTagg/otros/cpubkt 36/27/21/16 % de los cambios de proteccion).
// Cada falta: zona de la memoria (FrameNativo::Classify: DefaultTagg, cpuheap, cpubkt, ring, ctx, gpupage, otros), via
// del manejador (ventana, primera escritura, sin-vigilancia, escritura normal, lectura de la GPU, aplicada diferida...),
// ciclos dentro del manejador, hilo (el de envio = el que llama a sceAgcSuspendPoint, o los demas) y, 1 de cada 4, la
// instruccion que escribe (offset en el eboot; codigo del emulador -> la direccion de vuelta de la pila si es del eboot).
// Camino critico: el tiempo que el hilo de envio espera dentro de sceAgcSuspendPoint (GuestGpu::Done: espera al
// emulador). Si esa espera es mayor que sus faltas, las faltas del juego NO retrasan el envio del fotograma.
// Linea "[techo-faltas]" cada 5 s (desde Techo::Report, hilo de render).
#include "local/techo.h"
#include "local/frame-nativo.h"
#include "local/faltas-reciclado.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace TechoFaltas {

// Via del manejador (la pone HandleFault en el hilo que falla).
enum Via : uint8_t { VDiferida = 0, VPrimera, VSinVigilar, VVentana, VRestaurada, VEscritura, VLectura, VNoMapeada, Vias };
inline const char* const kVias[Vias] = {"diferida", "primera", "sin-vigilar", "ventana", "restaurada", "escritura", "lectura", "no-mapeada"};
inline thread_local uint8_t t_via      = VNoMapeada;
inline thread_local bool    t_envio    = false; // este hilo llama a sceAgcSuspendPoint
inline void                 SetVia(Via v) noexcept { t_via = v; }

// (faltas2) KYTY_FALTAS_TEX: la falta de escritura normal (via "escritura") no toma el cerrojo de la cache de texturas
// cuando el mapa de granulos de 64 KiB dice que alli no hay ninguna imagen (TextureCache::MayHaveImagesRead, el mismo
// atajo que IsRegionGpuModified). =1 encendido, =2 A/B dentro de la partida (cambia cada ventana de 5 s de techo),
// sin variable o =0 como antes. Seguro: una imagen que se registre despues protege la pagina y la escritura vuelve a
// fallar por el camino normal.
inline std::atomic<int> g_tex_mode {-1};
inline std::atomic<bool> g_tex_now {false};
[[nodiscard]] inline int TexMode() noexcept {
	int m = g_tex_mode.load(std::memory_order_relaxed);
	if (m >= 0) return m;
	const char* v = std::getenv("KYTY_FALTAS_TEX");
	m             = (v != nullptr && (v[0] == '1' || v[0] == '2')) ? v[0] - '0' : 0;
	g_tex_mode.store(m, std::memory_order_relaxed);
	g_tex_now.store(m == 1, std::memory_order_relaxed);
	std::printf("[techo-faltas] KYTY_FALTAS_TEX=%d: %s\n", m,
	            m == 1 ? "encendido (falta de escritura sin cerrojo de texturas donde no hay imagenes)"
	                   : m == 2 ? "A/B: cambia cada 5 s" : "apagado (como antes)");
	std::fflush(stdout);
	return m;
}
[[nodiscard]] inline bool TexSkip() noexcept {
	return TexMode() != 0 && g_tex_now.load(std::memory_order_relaxed);
}
// Tramos de la via "escritura": [0] ventana intentada, [1] RestoreIfUnwatched, [2] BufferCache::InvalidateMemory
// (incluye la lectura de vuelta de la GPU si la pagina era suya), [3] TextureCache::InvalidateMemory. Ciclos y llamadas.
inline std::atomic<uint64_t> g_tramo[4] {};
inline std::atomic<uint64_t> g_tramo_n {0}, g_tex_skipped {0};
inline std::atomic<uint64_t> g_lect_cyc {0}; // lectura: BufferCache::ReadMemory
// (sync1) SendCommandSync de hilos que no son el de la GPU: espera hasta que el hilo de la GPU lo empieza (su turno) y
// lo que dura el comando (DiscardGpuWrites, ReadMemory...). Ciclos y llamadas.
inline std::atomic<uint64_t> g_sync_turn {0}, g_sync_run {0}, g_sync_n {0};
// (sync2) KYTY_SYNC_TURNO: con OTROS2 el preambulo de ProcessPm4 se salta pasos "que no lo necesitan" aunque haya
// comandos de la CPU en cola (SendCommandSync): esperan al siguiente paso no saltado. =1: con comandos en cola el
// preambulo corre (lo mismo que sin OTROS2); =2 A/B cada 5 s (KYTY_TECHO); sin variable o 0 como antes.
inline std::atomic<int>  g_turno_mode {-1};
inline std::atomic<bool> g_turno_now {false};
[[nodiscard]] inline int TurnoMode() noexcept {
	int m = g_turno_mode.load(std::memory_order_relaxed);
	if (m >= 0) return m;
	const char* v = std::getenv("KYTY_SYNC_TURNO");
	// (sync3) Encendido por defecto: con comandos en cola el preambulo corre, lo mismo que sin OTROS2 (comportamiento de
	// antes en esos pasos); T54 A/B -0.29 ms/fot. =0 lo apaga, =2 A/B.
	m             = (v != nullptr && v[0] == '0') ? 0 : (v != nullptr && v[0] == '2') ? 2 : 1;
	g_turno_now.store(m == 1, std::memory_order_relaxed);
	g_turno_mode.store(m, std::memory_order_relaxed);
	std::printf("[techo-faltas] KYTY_SYNC_TURNO=%d: %s\n", m, m == 1 ? "encendido" : m == 2 ? "A/B cada 5 s" : "apagado (como antes)");
	std::fflush(stdout);
	return m;
}


constexpr uint32_t Zonas = FrameNativo::CCount;
// [hilo 0 otros / 1 envio][zona][via]: ciclos y llamadas.
inline std::atomic<uint64_t> g_cyc[2][Zonas][Vias] {};
inline std::atomic<uint64_t> g_n[2][Zonas][Vias] {};
inline std::atomic<uint64_t> g_wait_cyc {0}, g_wait_n {0}; // espera del hilo de envio en sceAgcSuspendPoint

// Sitios: clave = offset en el eboot (+1; bit 40: via codigo del emulador) | zona << 48. Tabla abierta, muestreada.
constexpr size_t Slots = 2048;
struct Slot {
	std::atomic<uint64_t> key {0}, cyc {0}, n {0}, envio {0};
};
inline Slot                  g_sites[Slots];
inline std::atomic<uint64_t> g_lost {0};
inline std::atomic<uint64_t> g_sample {0};

[[nodiscard]] inline bool SampleSite() noexcept {
	return (g_sample.fetch_add(1, std::memory_order_relaxed) & 3u) == 0;
}

// `site`: offset en el eboot + 1 (0: desconocido), con el bit 40 si la instruccion era del emulador (memcpy del runtime).
inline void Note(uint64_t vaddr, uint64_t cycles, uint64_t site, bool sampled) noexcept {
	const uint32_t z = FrameNativo::Classify(vaddr);
	const uint32_t v = t_via < Vias ? t_via : VNoMapeada;
	const uint32_t h = t_envio ? 1 : 0;
	g_cyc[h][z][v].fetch_add(cycles, std::memory_order_relaxed);
	g_n[h][z][v].fetch_add(1, std::memory_order_relaxed);
	if (!sampled) return;
	const uint64_t key = (site | (uint64_t {z} << 48u) | (uint64_t {v} << 52u)) + 1u;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 53u) % Slots;
	for (size_t p = 0; p < 32; ++p, i = (i + 1) % Slots) {
		uint64_t cur = g_sites[i].key.load(std::memory_order_relaxed);
		if (cur == 0 && g_sites[i].key.compare_exchange_strong(cur, key, std::memory_order_relaxed)) cur = key;
		if (cur != key) continue;
		g_sites[i].cyc.fetch_add(cycles, std::memory_order_relaxed);
		g_sites[i].n.fetch_add(1, std::memory_order_relaxed);
		if (h != 0) g_sites[i].envio.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	g_lost.fetch_add(1, std::memory_order_relaxed);
}

// sceAgcSuspendPoint: marca el hilo de envio y mide su espera al emulador.
struct EnvioScope {
	uint64_t t0;
	EnvioScope(): t0(Techo::On() ? __rdtsc() : 0) {
		if (t0 != 0) t_envio = true;
	}
	~EnvioScope() {
		if (t0 == 0) return;
		g_wait_cyc.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
		g_wait_n.fetch_add(1, std::memory_order_relaxed);
	}
	EnvioScope(const EnvioScope&)            = delete;
	EnvioScope& operator=(const EnvioScope&) = delete;
};

// Desde Techo::Report (cada 5 s): `frames` fotogramas, `ms_cyc` ms por ciclo TSC.
inline void Report(double frames, double ms_cyc) noexcept {
	if (frames <= 0) return;
	static uint64_t last_cyc[2][Zonas][Vias] {}, last_n[2][Zonas][Vias] {}, last_wait = 0, last_wait_n = 0;
	const auto      ms = [&](uint64_t c) { return static_cast<double>(c) * ms_cyc / frames; };
	double          zona_ms[Zonas] {}, zona_n[Zonas] {}, via_ms[Vias] {}, via_n[Vias] {}, hilo_ms[2] {}, hilo_n[2] {};
	for (uint32_t h = 0; h < 2; ++h)
		for (uint32_t z = 0; z < Zonas; ++z)
			for (uint32_t v = 0; v < Vias; ++v) {
				const uint64_t c = g_cyc[h][z][v].load(std::memory_order_relaxed), n = g_n[h][z][v].load(std::memory_order_relaxed);
				const double   dm = ms(c - last_cyc[h][z][v]), dn = static_cast<double>(n - last_n[h][z][v]) / frames;
				last_cyc[h][z][v] = c, last_n[h][z][v] = n;
				zona_ms[z] += dm, zona_n[z] += dn, via_ms[v] += dm, via_n[v] += dn, hilo_ms[h] += dm, hilo_n[h] += dn;
			}
	const uint64_t w = g_wait_cyc.load(std::memory_order_relaxed), wn = g_wait_n.load(std::memory_order_relaxed);
	const double   wait_ms = ms(w - last_wait), wait_n = static_cast<double>(wn - last_wait_n) / frames;
	last_wait = w, last_wait_n = wn;
	std::printf("[techo-faltas] 5 s: faltas %.2f ms/fot (%.0f/fot), hilo de envio %.2f ms (%.0f) + otros %.2f ms (%.0f); "
	            "espera del hilo de envio en sceAgcSuspendPoint %.2f ms/fot (%.1f/fot: %s) | por zona:",
	            hilo_ms[0] + hilo_ms[1], hilo_n[0] + hilo_n[1], hilo_ms[1], hilo_n[1], hilo_ms[0], hilo_n[0], wait_ms, wait_n,
	            wait_ms > hilo_ms[1] ? "espera mas que sus faltas: las faltas no retrasan el envio" : "sus faltas pueden retrasar el envio");
	for (uint32_t z = 0; z < Zonas; ++z)
		if (zona_n[z] >= 0.05) std::printf(" %s %.2f (%.0f)", FrameNativo::kNames[z], zona_ms[z], zona_n[z]);
	std::printf(" | por via:");
	for (uint32_t v = 0; v < Vias; ++v)
		if (via_n[v] >= 0.05) std::printf(" %s %.2f (%.0f)", kVias[v], via_ms[v], via_n[v]);
	// Sitios (muestreo 1/4, se escala x4): los 16 mas caros de la ventana.
	struct Row {
		uint64_t key, cyc, n, envio;
	};
	Row    top[16] {};
	size_t used = 0;
	for (auto& s: g_sites) {
		const uint64_t key = s.key.load(std::memory_order_relaxed);
		if (key == 0) continue;
		const Row r {key - 1u, s.cyc.exchange(0, std::memory_order_relaxed), s.n.exchange(0, std::memory_order_relaxed),
		             s.envio.exchange(0, std::memory_order_relaxed)};
		if (r.n == 0) continue;
		size_t k = used < 16 ? used++ : 16;
		if (k == 16) {
			if (r.cyc <= top[15].cyc) continue;
			k = 15;
		}
		while (k > 0 && top[k - 1].cyc < r.cyc) {
			top[k] = top[k - 1];
			--k;
		}
		top[k] = r;
	}
	std::printf(" | sitios (eboot+off zona via ms/fot faltas/fot, x4; 'emu<-' = memcpy del emulador llamado desde ahi):");
	for (size_t k = 0; k < used; ++k) {
		const uint64_t key  = top[k].key;
		const uint64_t site = key & ((uint64_t {1} << 40u) - 1u);
		const bool     emu  = ((key >> 40u) & 1u) != 0;
		const uint32_t z = static_cast<uint32_t>((key >> 48u) & 0xfu), v = static_cast<uint32_t>((key >> 52u) & 0xfu);
		if (site == 0) std::printf(" ?");
		else std::printf(" %s%llx", emu ? "emu<-" : "", static_cast<unsigned long long>(site - 1u));
		std::printf(" %s %s %.2f %.1f%s", z < Zonas ? FrameNativo::kNames[z] : "?", v < Vias ? kVias[v] : "?", ms(top[k].cyc) * 4.0,
		            static_cast<double>(top[k].n) * 4.0 / frames, top[k].envio != 0 ? " (envio)" : "");
		std::printf(k + 1 < used ? "," : "");
	}
	// Tramos de "escritura" y la A/B de KYTY_FALTAS_TEX.
	static uint64_t lt[4] {}, ltn = 0, lsk = 0, llc = 0;
	double          tr[4];
	for (int i = 0; i < 4; ++i) {
		const uint64_t c = g_tramo[i].load(std::memory_order_relaxed);
		tr[i] = ms(c - lt[i]), lt[i] = c;
	}
	const uint64_t tn = g_tramo_n.load(std::memory_order_relaxed), sk = g_tex_skipped.load(std::memory_order_relaxed),
	               lc = g_lect_cyc.load(std::memory_order_relaxed);
	const double esc_n = static_cast<double>(tn - ltn) / frames, sk_n = static_cast<double>(sk - lsk) / frames, lect = ms(lc - llc);
	ltn = tn, lsk = sk, llc = lc;
	static uint64_t lst = 0, lsr = 0, lsn = 0;
	const uint64_t  st = g_sync_turn.load(std::memory_order_relaxed), sr = g_sync_run.load(std::memory_order_relaxed),
	               sn = g_sync_n.load(std::memory_order_relaxed);
	static uint64_t lil = 0, lif = 0, lir = 0;
	const uint64_t  il = FaltasReciclado::g_inv_lock.load(), ifl = FaltasReciclado::g_inv_flush.load(),
	               ir = FaltasReciclado::g_inv_regions.load();
	std::printf(" | buffers por dentro: cerrojo de region %.2f ms (%.0f/fot), lectura/descarte %.2f ms, resto = buffers - ambos",
	            ms(il - lil), static_cast<double>(ir - lir) / frames, ms(ifl - lif));
	lil = il, lif = ifl, lir = ir;
	static uint64_t lrw = 0, lrc = 0, lrr = 0, lrn = 0;
	const uint64_t  rw = FaltasReciclado::g_rb_wait.load(), rc = FaltasReciclado::g_rb_copy.load(),
	               rr = FaltasReciclado::g_rb_release.load(), rn = FaltasReciclado::g_rb_n.load();
	std::printf(" | lectura de vuelta en el hilo del juego (%.0f/fot): espera a la GPU %.2f ms, copia %.2f ms, put-in de guardados %.2f ms",
	            static_cast<double>(rn - lrn) / frames, ms(rw - lrw), ms(rc - lrc), ms(rr - lrr));
	lrw = rw, lrc = rc, lrr = rr, lrn = rn;
	static uint64_t lwd = 0, lwl = 0;
	const uint64_t  wd = FaltasReciclado::g_win_dead.load(), wl = FaltasReciclado::g_win_live.load();
	std::printf(" (paginas de la ventana: muertas %.1f/fot, vivas %.1f/fot)", static_cast<double>(wd - lwd) / frames,
	            static_cast<double>(wl - lwl) / frames);
	lwd = wd, lwl = wl;
	std::printf(" | SendCommandSync (%.0f/fot): espera de turno %.2f ms, comando %.2f ms", static_cast<double>(sn - lsn) / frames,
	            ms(st - lst), ms(sr - lsr));
	lst = st, lsr = sr, lsn = sn;
	std::printf(" | escritura por tramo (%.0f/fot): ventana %.2f restaurar %.2f buffers %.2f texturas %.2f ms (sin cerrojo %.0f/fot);"
	            " lectura ReadMemory %.2f ms",
	            esc_n, tr[0], tr[1], tr[2], tr[3], sk_n, lect);
	const int mode = TexMode();
	if (mode == 2) {
		static double sum[2] {}, cnt[2] {};
		const bool    was = g_tex_now.load(std::memory_order_relaxed);
		sum[was ? 1 : 0] += hilo_ms[0] + hilo_ms[1];
		cnt[was ? 1 : 0] += 1;
		const double a = cnt[0] > 0 ? sum[0] / cnt[0] : 0, b = cnt[1] > 0 ? sum[1] / cnt[1] : 0;
		std::printf(" | A/B KYTY_FALTAS_TEX: esta ventana %s; media faltas A(antes) %.2f ms (%.0f v) B(nuevo) %.2f ms (%.0f v) -> %+.2f ms/fot",
		            was ? "B" : "A", a, cnt[0], b, cnt[1], (cnt[0] > 0 && cnt[1] > 0) ? b - a : 0.0);
		g_tex_now.store(!was, std::memory_order_relaxed);
	}
	if (TurnoMode() == 2) {
		static double sum[2] {}, cnt[2] {};
		const bool    was = g_turno_now.load(std::memory_order_relaxed);
		sum[was ? 1 : 0] += hilo_ms[0] + hilo_ms[1];
		cnt[was ? 1 : 0] += 1;
		const double a = cnt[0] > 0 ? sum[0] / cnt[0] : 0, b = cnt[1] > 0 ? sum[1] / cnt[1] : 0;
		std::printf(" | A/B KYTY_SYNC_TURNO: esta ventana %s; media faltas A %.2f ms (%.0f v) B %.2f ms (%.0f v) -> %+.2f ms/fot",
		            was ? "B" : "A", a, cnt[0], b, cnt[1], (cnt[0] > 0 && cnt[1] > 0) ? b - a : 0.0);
		g_turno_now.store(!was, std::memory_order_relaxed);
	}
	char rec[400];
	FaltasReciclado::Report(rec, sizeof(rec));
	std::printf("%s", rec);
	if (FaltasReciclado::Mode() == 2) {
		static double sum[2] {}, cnt[2] {};
		const bool    was = FaltasReciclado::g_now.load(std::memory_order_relaxed);
		sum[was ? 1 : 0] += hilo_ms[0] + hilo_ms[1];
		cnt[was ? 1 : 0] += 1;
		const double a = cnt[0] > 0 ? sum[0] / cnt[0] : 0, b = cnt[1] > 0 ? sum[1] / cnt[1] : 0;
		std::printf(" | A/B KYTY_FALTAS_RECICLADO: esta ventana %s; media faltas A(antes) %.2f ms (%.0f v) B(nuevo) %.2f ms (%.0f v) -> %+.2f ms/fot",
		            was ? "B" : "A", a, cnt[0], b, cnt[1], (cnt[0] > 0 && cnt[1] > 0) ? b - a : 0.0);
		FaltasReciclado::g_now.store(!was, std::memory_order_relaxed);
	}
	std::printf(" | perdidos %llu\n", static_cast<unsigned long long>(g_lost.exchange(0, std::memory_order_relaxed)));
	std::fflush(stdout);
}

} // namespace TechoFaltas
