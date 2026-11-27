#pragma once
// (sombras-ab) KYTY_NATIVO_SOMBRAS_AB=N (segundos, 1..600; 0 o sin poner: apagado). Medicion A/B en UN solo proceso de
// lo que dan las sombras nativas (KYTY_NATIVO_PASE_SOMBRAS=1): cada N s se alterna entre
//   "nativo": las sombras nativas hacen lo de siempre (reemplazan las cascadas ya validadas), y
//   "juego":  DsNatStart (renderer/dsSombrasNativas.inc) no toma ningun cubo: el juego dibuja todo, como sin =1, pero
//             el gancho sigue instalado (su coste en el hilo del juego esta en los dos tramos por igual).
// El modo solo cambia en el flip (LiveControl::Flip, hilo de render): un cubo ya empezado sigue con lo que eligio en
// DsNatStart, nunca queda a medias. Al final de cada tramo una linea "[sombras-ab] tramo nativo|juego"; al cerrar (y
// tras cada tramo, por si el proceso muere) "[sombras-ab] RESUMEN" con la media por modo y la diferencia en ms y %.
// Datos: los que ya existen en el flip: el tiempo de cada fotograma (flip a flip, steady clock), la espera del hilo de
// render por trabajo del juego (LiveControl::g_render_idle_ns: ocupado = fotograma - espera) y la espera por
// traduccion de shaders / pipelines (LiveCensus::g_waits_ns[WaitCompile]). Tiempo de GPU por fotograma: solo existe
// con KYTY_F0 (su propia linea [f0]); aqui no se inventa.
// El primer tramo (arranque, shaders aun compilandose) se imprime pero queda fuera del resumen.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace DsSombrasAb {

[[nodiscard]] inline uint32_t Seconds() {
	static const uint32_t s = [] {
		const char* text = std::getenv("KYTY_NATIVO_SOMBRAS_AB");
		const long  n    = text != nullptr ? std::strtol(text, nullptr, 10) : 0;
		const auto  v    = n >= 1 && n <= 600 ? static_cast<uint32_t>(n) : 0u;
		const char* pase = std::getenv("KYTY_NATIVO_PASE_SOMBRAS");
		if (v != 0) {
			std::printf("[sombras-ab] KYTY_NATIVO_SOMBRAS_AB=%u: alterna cada %u s nativo / juego (KYTY_NATIVO_PASE_SOMBRAS=%s)%s\n", v, v,
			            pase != nullptr ? pase : "(sin poner)",
			            pase != nullptr && pase[0] == '1' && pase[1] == '\0' ? "" : " - AVISO: sin =1 los dos tramos son iguales");
			std::fflush(stdout);
		}
		return v;
	}();
	return s;
}

// true: tramo "juego" (DsNatStart no toma cubos). Escrito solo en el flip.
inline std::atomic_bool g_juego {false};
[[nodiscard]] inline bool Juego() noexcept { return g_juego.load(std::memory_order_relaxed); }

// Tiempos del camino nativo (solo con KYTY_NATIVO_SOMBRAS_AB; apagado: una lectura de un static). ns acumulados en el
// tramo, por todos los hilos. Salen en la linea "[sombras-ab] tramo" en ms por fotograma.
enum Coste : uint32_t {
	CGancho = 0,  // gancho 0xab57d0 entero (hilo del juego, incluye la funcion original)
	COriginal,    // la funcion original del juego dentro del gancho
	CFill,        // recorrido de la DCB + datos por entrada (hilo del juego)
	CMarca,       // DsNatMark (hilo de render: marcas de inicio/fin, NatTake, agrupado)
	CAntesDraw,   // DsNatBeforeDraw entero (hilo de render; incluye CStart y CWork)
	CStart,       // DsNatStart (Lookups de MALLA-NATIVA, candados, IsMetaCleared)
	CWork,        // DsNatWork (ObtainBuffer del skinning, EndRendering, barreras, dibujo)
	CCubo,        // (T43) ProcessPm4 entre la marca de inicio y la de fin de cada cubo del sol (hilo de render, sin DsNatMark)
	CCount
};
inline std::atomic<uint64_t> g_coste_ns[CCount] {};
inline std::atomic<uint64_t> g_coste_n[CCount] {};
// (verificacion del arreglo del ES) Entradas emparejadas en la GPU: ES del gancho == ES real del draw / distinto / sin ES.
inline std::atomic<uint64_t> g_es_igual {0}, g_es_distinto {0}, g_es_sin {0};
[[nodiscard]] inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Timer {
	uint32_t k;
	int64_t  t0;
	explicit Timer(uint32_t which): k(which), t0(Seconds() != 0 ? NowNs() : 0) {}
	~Timer() {
		if (t0 == 0) return;
		g_coste_ns[k].fetch_add(static_cast<uint64_t>(NowNs() - t0), std::memory_order_relaxed);
		g_coste_n[k].fetch_add(1, std::memory_order_relaxed);
	}
	Timer(const Timer&)            = delete;
	Timer& operator=(const Timer&) = delete;
};

struct Modo {
	uint64_t            frames = 0;
	double              wall_ms = 0, busy_ms = 0, compile_ms = 0;
	std::vector<float>  frame_ms;
};
inline Modo g_total[2]; // [0 nativo, 1 juego], sin el primer tramo
inline uint32_t g_tramos[2] {};
// (T40 11-27: los fps subieron de 56 a 101 durante la partida; la media de cada modo no vale.) Cada tramo (sin el 1o):
// modo, fps y render ocupado por fotograma. El RESUMEN compara cada tramo con la media de sus dos vecinos (del otro modo).
struct Tramo { uint32_t modo; double fps, busy; };
inline std::vector<Tramo> g_serie;

[[nodiscard]] inline double P99(std::vector<float> v) {
	if (v.empty()) return 0.0;
	const size_t k = std::min(v.size() - 1, static_cast<size_t>(static_cast<double>(v.size()) * 0.99));
	std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
	return v[k];
}

inline void Resumen(const char* cuando) {
	const Modo& n = g_total[0];
	const Modo& j = g_total[1];
	if (n.frames == 0 || j.frames == 0) {
		std::printf("[sombras-ab] RESUMEN (%s): faltan tramos completos de los dos modos (nativo %u, juego %u)\n", cuando, g_tramos[0], g_tramos[1]);
		std::fflush(stdout);
		return;
	}
	const auto fps   = [](const Modo& m) { return static_cast<double>(m.frames) * 1e3 / m.wall_ms; };
	const auto frame = [](const Modo& m) { return m.wall_ms / static_cast<double>(m.frames); };
	const auto busy  = [](const Modo& m) { return m.busy_ms / static_cast<double>(m.frames); };
	const auto comp  = [](const Modo& m) { return m.compile_ms / static_cast<double>(m.frames); };
	const double fn = frame(n), fj = frame(j), bn = busy(n), bj = busy(j);
	const double pn = P99(n.frame_ms), pj = P99(j.frame_ms);
	std::printf("[sombras-ab] RESUMEN (%s; sin el 1er tramo): nativo %u tramos %llu frames %.1f fps, fotograma %.2f ms, render ocupado %.2f ms "
	            "(traduccion %.2f), p99 %.2f ms | juego %u tramos %llu frames %.1f fps, fotograma %.2f ms, render ocupado %.2f ms "
	            "(traduccion %.2f), p99 %.2f ms | nativo - juego: fotograma %+.2f ms (%+.1f%%), fps %+.1f (%+.1f%%), render ocupado %+.2f ms "
	            "(%+.1f%%), p99 %+.2f ms (%+.1f%%) | GPU: no medido aqui (solo con KYTY_F0)\n",
	            cuando, g_tramos[0], static_cast<unsigned long long>(n.frames), fps(n), fn, bn, comp(n), pn, g_tramos[1],
	            static_cast<unsigned long long>(j.frames), fps(j), fj, bj, comp(j), pj, fn - fj, fj > 0 ? (fn - fj) * 100.0 / fj : 0.0,
	            fps(n) - fps(j), fps(j) > 0 ? (fps(n) - fps(j)) * 100.0 / fps(j) : 0.0, bn - bj, bj > 0 ? (bn - bj) * 100.0 / bj : 0.0,
	            pn - pj, pj > 0 ? (pn - pj) * 100.0 / pj : 0.0);
	// Interpolado entre vecinos: tramo i con i-1 e i+1 del otro modo. nativo - juego (%), media de todos los tramos validos.
	double   sf = 0, sb = 0;
	uint32_t k  = 0;
	for (size_t i = 1; i + 1 < g_serie.size(); ++i) {
		const Tramo &a = g_serie[i - 1], &c = g_serie[i], &b = g_serie[i + 1];
		if (a.modo == c.modo || b.modo == c.modo) continue;
		const double vf = (a.fps + b.fps) * 0.5, vb = (a.busy + b.busy) * 0.5;
		if (vf <= 0 || vb <= 0 || c.fps <= 0 || c.busy <= 0) continue;
		const double df = (c.fps - vf) * 100.0 / vf, db = (c.busy - vb) * 100.0 / vb;
		sf += c.modo == 0 ? df : -df;
		sb += c.modo == 0 ? db : -db;
		++k;
	}
	if (k != 0)
		std::printf("[sombras-ab] RESUMEN interpolado (%s; cada tramo frente a la media de sus 2 vecinos del otro modo, %u tramos): "
		            "nativo - juego: fps %+.1f%%, render ocupado %+.1f%%\n",
		            cuando, k, sf / k, sb / k);
	std::fflush(stdout);
}

// Hilo de render, en cada flip (LiveControl::Flip). idle_ns / compile_ns: acumulados desde el arranque.
inline void OnFlip(int64_t flip_ns, int64_t idle_ns, int64_t compile_ns) {
	const uint32_t secs = Seconds();
	if (secs == 0) return;
	static int64_t  start_ns = 0, last_ns = 0, last_idle = 0, last_compile = 0;
	static uint32_t tramo = 0;
	static Modo     cur;
	if (last_ns == 0) {
		static const bool registrado = [] { return std::atexit([] { Resumen("al cerrar"); }) == 0; }();
		(void)registrado;
		start_ns = last_ns = flip_ns, last_idle = idle_ns, last_compile = compile_ns;
		g_juego.store(false, std::memory_order_relaxed);
		return;
	}
	const double ms   = static_cast<double>(flip_ns - last_ns) / 1e6;
	const double idle = static_cast<double>(idle_ns - last_idle) / 1e6;
	cur.frames++;
	cur.wall_ms += ms;
	cur.busy_ms += std::max(0.0, ms - idle);
	cur.compile_ms += static_cast<double>(compile_ns - last_compile) / 1e6;
	cur.frame_ms.push_back(static_cast<float>(ms));
	last_ns = flip_ns, last_idle = idle_ns, last_compile = compile_ns;
	if (flip_ns - start_ns < static_cast<int64_t>(secs) * 1000000000LL) return;
	// Fin de tramo (en el limite de fotograma): linea, acumulado y cambio de modo para el siguiente fotograma.
	const uint32_t modo = Juego() ? 1u : 0u;
	const double   n    = static_cast<double>(cur.frames);
	std::printf("[sombras-ab] tramo %s #%u%s: frames %llu, fps %.1f, fotograma %.2f ms, render ocupado %.2f ms (traduccion %.2f), "
	            "GPU n/d, p99 %.2f ms\n",
	            modo == 0 ? "nativo" : "juego", tramo, tramo == 0 ? " (calentamiento, fuera del resumen)" : "",
	            static_cast<unsigned long long>(cur.frames), n * 1e3 / cur.wall_ms, cur.wall_ms / n, cur.busy_ms / n,
	            cur.compile_ms / n, P99(cur.frame_ms));
	{
		static const char* const names[CCount] = {"gancho", "original del juego", "recorrido DCB", "DsNatMark", "DsNatBeforeDraw",
		                                          "DsNatStart", "DsNatWork", "cubos del sol en ProcessPm4"};
		char   line[1024];
		size_t at = 0;
		for (uint32_t c = 0; c < CCount && at < sizeof(line); ++c) {
			const uint64_t ns = g_coste_ns[c].exchange(0, std::memory_order_relaxed);
			const uint64_t cn = g_coste_n[c].exchange(0, std::memory_order_relaxed);
			at += static_cast<size_t>(std::snprintf(line + at, sizeof(line) - at, "%s%s %.3f ms/fot (%llu llamadas, %.1f us c/u)",
			                                        c == 0 ? "" : ", ", names[c], static_cast<double>(ns) / 1e6 / n,
			                                        static_cast<unsigned long long>(cn),
			                                        cn != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(cn) : 0.0));
		}
		const uint64_t ei = g_es_igual.exchange(0), ed = g_es_distinto.exchange(0), es = g_es_sin.exchange(0);
		std::printf("[sombras-ab] coste %s #%u (gancho y recorrido: hilos del juego, en paralelo; el resto: hilo de render): %s | "
		            "ES gancho vs draw real: igual=%llu distinto=%llu sin ES=%llu\n",
		            modo == 0 ? "nativo" : "juego", tramo, line, static_cast<unsigned long long>(ei), static_cast<unsigned long long>(ed),
		            static_cast<unsigned long long>(es));
	}
	if (tramo != 0) {
		Modo& t = g_total[modo];
		t.frames += cur.frames, t.wall_ms += cur.wall_ms, t.busy_ms += cur.busy_ms, t.compile_ms += cur.compile_ms;
		t.frame_ms.insert(t.frame_ms.end(), cur.frame_ms.begin(), cur.frame_ms.end());
		g_tramos[modo]++;
		g_serie.push_back({modo, n * 1e3 / cur.wall_ms, cur.busy_ms / n});
		Resumen("parcial");
	} else {
		std::fflush(stdout);
	}
	cur = Modo {};
	++tramo;
	start_ns = flip_ns;
	g_juego.store(modo == 0, std::memory_order_relaxed);
}

} // namespace DsSombrasAb
