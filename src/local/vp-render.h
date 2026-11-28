#pragma once
// (vp-render) Dos cosas del hilo de render que valen para todo el juego (con o sin KYTY_TEJ):
//
// 1) KYTY_VP_RENDER (1 encendido por defecto; =0 apagado; =2 A/B cada 5 s con KYTY_TECHO=1): las LIBERACIONES DE
//    ESCRITURA del hilo de render (UpdateRegionWatchers, origen write-release: solo lectura -> lectura/escritura) se
//    dejan al hilo de liberacion, igual que KYTY_RAIZ_ASYNC_RELEASE (encendido por defecto) ya hace con las de lectura:
//    la pagina se marca (Region::deferred), su estado ya dice liberada y el host sigue en solo lectura (MAS estricto)
//    hasta que la aplica el hilo de liberacion, el hilo del juego que la espera o una falta sobre ella
//    (GpuResourceManager::HandleFault -> ApplyDeferredReleaseAt, lo primero: aplica la proteccion de su estado y
//    reintenta). Una liberacion solo quita proteccion: tarde no esconde ninguna escritura (la escritura falla y la aplica
//    antes). Una proteccion posterior del gestor sobre esas paginas borra la marca (ClearDeferredAfterProtect) y la
//    aplicacion usa SIEMPRE el estado actual de la pagina (PermsOf bajo su cerrojo): nunca deja una proteccion vieja.
//    Log 20261128-104552-500 ([raiz] PROTECT BY ORIGIN, hilo GPU por fot): write-release 290 llamadas 0.44 ms,
//    read-flush 148 / 0.76 ms, write-track 50 / 0.10 ms; LAG4: 394-490 VirtualProtect/fot, 1.28-1.58 ms/fot.
//    No es identico al 100 % (una escritura del juego antes de que se aplique cuesta una falta en SU hilo, contada en
//    [raiz] ASYNC RELEASE "fault"): por eso apagado por defecto hasta medirlo.
// 2) KYTY_RENDER_IDEAL (1 encendido por defecto; =0 apagado; =2 A/B): procesador IDEAL (SetThreadIdealProcessorEx: solo una
//    preferencia, no afinidad dura) del hilo de render: un nucleo P (hibridos), de la L3 grande (X3D), nunca el nucleo
//    del CPU 0 (interrupciones), el primer hilo SMT del nucleo. Con KYTY_RENDER_CPUS (afinidad dura) no se toca.
//    LAG4: el render migra 1.66-2.83 veces/fot (aparece en los nucleos 0 y 11; KYTY_VCACHE_PIN=0 en ese log).
// Linea [vp-render] por ventana: llamadas/ms de VirtualProtect del render por lado, migraciones, liberaciones
// diferidas, y el delta interpolado B-A de render ocupado (patron 0f9920d3).
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/techo.h"

namespace VpRender {

[[nodiscard]] inline int ModeOf(const char* name) {
	// (28/11) ENCENDIDO POR DEFECTO: medido en el PC del usuario (log 20261128-105930, build 9378cf3): VirtualProtect
	// del render 1.28-1.58 -> 0.83-0.94 ms/fot y migraciones 1.66-2.83 -> 1.17-1.48/fot, sin errores. =0 lo apaga.
	const char* v = std::getenv(name);
	int         m = v != nullptr ? std::atoi(v) : 1;
	if (m != 1 && m != 2) m = 0;
	if (m == 2 && !Techo::On()) m = 1; // (A/B solo con KYTY_TECHO=1; sin el, encendido)
	return m;
}
[[nodiscard]] inline int DeferMode() {
	static const int m = ModeOf("KYTY_VP_RENDER");
	return m;
}
[[nodiscard]] inline int IdealMode() {
	static const int m = ModeOf("KYTY_RENDER_IDEAL");
	return m;
}

inline std::atomic<bool> g_ab_on {false}; // (=2) esta ventana es B (encendido); todos los A/B alternan juntos

[[nodiscard]] inline bool DeferWriteRelease() {
	const int m = DeferMode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}
[[nodiscard]] inline bool IdealWanted() {
	const int m = IdealMode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

// (pageManager.cpp, Windows) Pone / quita el procesador ideal del hilo que llama. Devuelve el CPU elegido o -1.
int ApplyIdeal(bool on);

struct alignas(64) Side {
	std::atomic<uint64_t> calls {0}, ticks {0}, mig {0}, def_runs {0}, def_pages {0};
};
inline Side                  g_side[2];
inline std::atomic<uint32_t> g_last_cpu {0xffffffffu};
inline std::atomic<int>      g_ideal_cpu {-1};
inline thread_local int      t_ideal_applied = -1; // (hilo de render: 0/1 lo aplicado)

[[nodiscard]] inline int SideNow() { return g_ab_on.load(std::memory_order_relaxed) ? 1 : 0; }

// (Hilo de render, cada VirtualProtect de ProtectHost.) Procesador ideal al dia; migraciones entre protecciones.
inline void OnRenderProtect(uint32_t cpu) {
	if (IdealMode() != 0) {
		const int want = IdealWanted() ? 1 : 0;
		if (t_ideal_applied != want) {
			t_ideal_applied = want;
			const int c     = ApplyIdeal(want != 0);
			if (want != 0) g_ideal_cpu.store(c, std::memory_order_relaxed);
		}
	}
	if (!Techo::On()) return;
	const uint32_t last = g_last_cpu.exchange(cpu, std::memory_order_relaxed);
	if (last != 0xffffffffu && last != cpu) g_side[SideNow()].mig.fetch_add(1, std::memory_order_relaxed);
}
inline void NoteProtect(uint64_t ticks) {
	auto& s = g_side[SideNow()];
	s.calls.fetch_add(1, std::memory_order_relaxed);
	s.ticks.fetch_add(ticks, std::memory_order_relaxed);
}
inline void NoteDeferred(uint64_t pages) {
	if (!Techo::On()) return;
	auto& s = g_side[SideNow()];
	s.def_runs.fetch_add(1, std::memory_order_relaxed);
	s.def_pages.fetch_add(pages, std::memory_order_relaxed);
}

// (techo-report, hilo de render, una vez por ventana de 5 s.) ms_tick: ms por tick del TSC.
inline void Window(double busy_ms, double fps, double frames, bool keep, double ms_tick) {
	const int dm = DeferMode(), im = IdealMode();
	if (!Techo::On() || frames <= 0) return;
	const bool was = g_ab_on.load(std::memory_order_relaxed);
	const int  sd  = was ? 1 : 0;
	auto&      s   = g_side[sd];
	const auto x   = [frames](std::atomic<uint64_t>& a) { return static_cast<double>(a.exchange(0, std::memory_order_relaxed)) / frames; };
	const double calls = x(s.calls), ticks = x(s.ticks), mig = x(s.mig), druns = x(s.def_runs), dpages = x(s.def_pages);
	const bool   ab    = dm == 2 || im == 2;
	double       delta = 0, cnt[2] {};
	if (ab) {
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
	std::printf("[vp-render] ventana %s (KYTY_VP_RENDER=%d KYTY_RENDER_IDEAL=%d, ideal CPU %d): render ocupado %.2f ms, "
	            "%.1f fps | VirtualProtect del render %.1f/fot %.3f ms/fot (%.2f us) | write-release diferidas %.1f/fot (%.0f "
	            "pag) | migraciones del render %.2f/fot",
	            ab ? (was ? "B (encendido)" : "A (apagado)") : (dm != 0 || im != 0 ? "encendido" : "apagado"), dm, im,
	            g_ideal_cpu.load(std::memory_order_relaxed), busy_ms, fps, calls, ticks * ms_tick,
	            calls > 0 ? ticks * ms_tick * 1e3 / calls : 0.0, druns, dpages, mig);
	if (ab)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (negativo = ahorro)", cnt[0], cnt[1], delta);
	std::printf("\n");
	std::fflush(stdout);
	if (ab) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace VpRender
