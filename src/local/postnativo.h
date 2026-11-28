#pragma once
// (postnativo) KYTY_POSTNATIVO (apagado por defecto; necesita KYTY_TECHO=1). SOLO MIDE en esta build.
// El tramo del postproceso de Demon's Souls 01.007.000 dentro de DCB#1 0xa150fa (FUN_00a14886), separado del resto.
//
// Ghidra (eboot 01.007.000): FUN_009eb6a0 abre el ambito del perfilador "PostProcessing" (ScopeBegin 0x999220 ->
// 0x99a080 -> 0x1d0a920 -> AgcCbReleaseMem 0x28 BOTTOM_OF_PIPE_TS al registro del ambito) y llama a FUN_009e8630:
//   PrepareMotionBlur 0x9e7b80, ComputeExposure 0x9e5570, SuperResolution 0xa52eb0 (si FUN_00a53700) o TemporalAA
//   0x9e8800, PreparePostBuffer 0x9eecc0, PrepareBloom 0x9ef900, ApplyDOF 0x9e8cd0, ApplyMotionBlur 0x9eb1d0,
//   ApplyPostFX 0x9e6520. Cada uno abre su propio ambito con su nombre (FUN_00999220(&s, ctx, 0x17, "Nombre")).
// El mapa del frame (src/native, KYTY_NATIVE_RENDER=map) ya sigue esos ambitos en la cola: aqui se enciende solo
// (si KYTY_NATIVE_RENDER no esta puesto) y, en cada draw / dispatch con "PostProcessing" abierto en la cola grafica,
// se cuentan: draws, dispatches, pipelines distintos (direcciones de VS/GS/PS o CS), destinos distintos (CB0 / Z),
// escrituras de registros SH (user data = punteros a constantes y descriptores) y una huella del fotograma
// (secuencia de opcode + shaders + destinos). Huella igual a la del fotograma anterior = el tramo seria reproducible
// tal cual; distinta = cambio (resolucion, SR on/off, menu, pelicula).
// Linea "[postnativo]" cada 5 s (ventana de [techo]): ms/fot del hilo GPU dentro del tramo (inclusivo, del mapa) y por
// sub-pase, y los contadores. La via nativa (=1/=2/=3) NO esta en esta build: falta el dato de estabilidad y las
// imagenes leidas (descriptores en memoria) para llegar al 90 % de seguridad.
// Coste apagado: una carga de un static por paquete de draw/dispatch.
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "local/techo.h"
#include "native/pass-cut.h"
#include "native/pass-registry.h"

namespace PostNativo {

[[nodiscard]] inline int Requested() {
	static const int r = [] {
		const char* v = std::getenv("KYTY_POSTNATIVO");
		const int   m = v != nullptr ? std::atoi(v) : 0;
		return m >= 1 && m <= 3 ? m : 0;
	}();
	return r;
}

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		int m = Requested();
		if (m != 0 && !Techo::On()) {
			std::printf("[postnativo] KYTY_POSTNATIVO=%d sin KYTY_TECHO=1: apagado\n", m);
			m = 0;
		}
		if (m != 0)
			std::printf("[postnativo] KYTY_POSTNATIVO=%d: contador del tramo del postproceso (ambito PostProcessing del "
			            "juego, DCB#1 a150fa)%s\n",
			            m, m == 1 ? "" : "; la via nativa (=2/=3) no esta en esta build: solo cuenta");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

constexpr int         Subs = 10;
inline const char*    kSub[Subs] = {"PostProcessing",    "PrepareMotionBlur", "ComputeExposure", "SuperResolution",
                                    "TemporalAA",        "PreparePostBuffer", "PrepareBloom",    "ApplyDOF",
                                    "ApplyMotionBlur",   "ApplyPostFX"};

// Todo en el hilo GPU (el que traduce); el informe lee los atomicos.
struct State {
	uint32_t              id[Subs];
	bool                  resolved = false;
	uint32_t              resolve_tries = 0;
	// fotograma en curso
	uint64_t              hash = 0xcbf29ce484222325ull;
	uint32_t              draws = 0, dispatches = 0, sh_writes = 0;
	uint32_t              sub_draws[Subs] {};
	std::vector<uint64_t> pipes, targets;
	uint64_t              prev_hash = 0;
	bool                  have_prev = false;
};
inline State* g_state = nullptr;

inline std::atomic<uint64_t> g_frames {0}, g_frames_post {0}, g_same {0}, g_diff {0}, g_draws {0}, g_disp {0},
    g_sh {0}, g_pipes {0}, g_targets {0}, g_sub_draws[Subs] {};

inline uint64_t Fold(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }

inline bool Resolve(State& s) {
	if (s.resolved && (++s.resolve_tries & 4095u) != 0u) return true;
	if (!s.resolved && (!NativeRender::PassRegistry::Ready() || (++s.resolve_tries & 255u) != 1u)) return false;
	if (!s.resolved)
		for (auto& i : s.id) i = NativeRender::NoPass;
	// (los nombres que aun no aparecieron se buscan de nuevo cada 4096 llamadas: el registro los anade al verlos)
	for (int k = 0; k < Subs; ++k) {
		if (s.id[k] != NativeRender::NoPass) continue;
		for (uint32_t i = 0, n = NativeRender::PassRegistry::Count(); i < n; ++i)
			if (NativeRender::PassRegistry::SameName(NativeRender::PassRegistry::Name(i), kSub[k])) {
				s.id[k] = i;
				break;
			}
	}
	s.resolved = s.id[0] != NativeRender::NoPass;
	return s.resolved;
}

// Un draw (draw = true) o dispatch de la cola que se traduce. vs/ps/cs: direcciones de programa; cb0/z: destinos.
namespace Luz {
inline void Work(uint32_t, bool, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
inline void Flip();
inline void Window(double, double);
} // namespace Luz
inline void Work(uint32_t opcode, bool draw, uint64_t vs, uint64_t ps, uint64_t cs, uint64_t cb0, uint64_t z) {
	Luz::Work(opcode, draw, vs, ps, cs, cb0, z); // (luz)
	if (!NativeRender::Cut::OnGpuThread() || NativeRender::Cut::Queue() != 0) return;
	if (g_state == nullptr) g_state = new State;
	State& s = *g_state;
	if (!Resolve(s) || !NativeRender::Cut::IsOpen(s.id[0])) return;
	if (draw) ++s.draws; else ++s.dispatches;
	const uint64_t pipe = draw ? Fold(Fold(vs, ps), 1) : Fold(cs, 2);
	s.hash = Fold(Fold(Fold(s.hash, opcode), pipe), draw ? Fold(cb0, z) : 0);
	s.pipes.push_back(pipe);
	if (draw) s.targets.push_back(Fold(cb0, z));
	const uint32_t inner = NativeRender::Cut::Innermost();
	for (int k = 1; k < Subs; ++k)
		if (s.id[k] != NativeRender::NoPass && s.id[k] == inner) {
			++s.sub_draws[k];
			break;
		}
}

// SET_SH_REG / SET_SH_REG_INDIRECT dentro del tramo (user data: punteros a constantes / descriptores).
inline void ShWrite() {
	if (g_state == nullptr || !g_state->resolved || !NativeRender::Cut::OnGpuThread() || NativeRender::Cut::Queue() != 0 ||
	    !NativeRender::Cut::IsOpen(g_state->id[0]))
		return;
	++g_state->sh_writes;
}

[[nodiscard]] inline size_t Distinct(std::vector<uint64_t>& v) {
	std::sort(v.begin(), v.end());
	return static_cast<size_t>(std::unique(v.begin(), v.end()) - v.begin());
}

// CommandProcessor::Flip (hilo GPU).
inline void Flip() {
	Luz::Flip(); // (luz)
	g_frames.fetch_add(1, std::memory_order_relaxed);
	if (g_state == nullptr) return;
	State& s = *g_state;
	if (s.draws + s.dispatches != 0) {
		g_frames_post.fetch_add(1, std::memory_order_relaxed);
		if (s.have_prev) (s.hash == s.prev_hash ? g_same : g_diff).fetch_add(1, std::memory_order_relaxed);
		s.prev_hash = s.hash, s.have_prev = true;
		g_draws.fetch_add(s.draws, std::memory_order_relaxed);
		g_disp.fetch_add(s.dispatches, std::memory_order_relaxed);
		g_sh.fetch_add(s.sh_writes, std::memory_order_relaxed);
		g_pipes.fetch_add(Distinct(s.pipes), std::memory_order_relaxed);
		g_targets.fetch_add(Distinct(s.targets), std::memory_order_relaxed);
		for (int k = 1; k < Subs; ++k) g_sub_draws[k].fetch_add(s.sub_draws[k], std::memory_order_relaxed);
	} else {
		s.have_prev = false; // (fotograma sin postproceso: menu / pelicula)
	}
	s.hash = 0xcbf29ce484222325ull;
	s.draws = s.dispatches = s.sh_writes = 0;
	for (auto& d : s.sub_draws) d = 0;
	s.pipes.clear();
	s.targets.clear();
}

// Ventana de 5 s de [techo] (hilo del informe). ms: tiempo inclusivo del hilo GPU con el ambito abierto (mapa).
inline void Window(double busy_ms, double /*fps*/, double frames, bool /*keep*/) {
	Luz::Window(busy_ms, frames); // (luz)
	if (Mode() == 0 || frames <= 0 || g_state == nullptr || !g_state->resolved) return;
	static int64_t last[Subs] {};
	double         ms[Subs] {};
	for (int k = 0; k < Subs; ++k) {
		if (g_state->id[k] == NativeRender::NoPass) continue;
		const int64_t now = NativeRender::Cut::InclNs(g_state->id[k]);
		ms[k]   = static_cast<double>(now - last[k]) * 1e-6 / frames;
		last[k] = now;
	}
	const double fp = static_cast<double>(g_frames_post.exchange(0, std::memory_order_relaxed));
	const double fa = static_cast<double>(g_frames.exchange(0, std::memory_order_relaxed));
	const double d  = fp > 0 ? fp : 1;
	const double same = static_cast<double>(g_same.exchange(0, std::memory_order_relaxed));
	const double diff = static_cast<double>(g_diff.exchange(0, std::memory_order_relaxed));
	std::printf("[postnativo] tramo PostProcessing: %.3f ms/fot de hilo GPU (render ocupado %.2f) | fot con post %.0f/%.0f | "
	            "draws %.1f disp %.1f pipelines distintos %.1f destinos distintos %.1f escrituras SH (user data) %.1f /fot | "
	            "huella igual al fot anterior %.0f, DIFERENTE %.0f (%.1f %% igual)",
	            ms[0], busy_ms, fp, fa, g_draws.exchange(0) / d, g_disp.exchange(0) / d, g_pipes.exchange(0) / d,
	            g_targets.exchange(0) / d, g_sh.exchange(0) / d, same, diff,
	            same + diff > 0 ? 100.0 * same / (same + diff) : 0.0);
	std::printf(" | sub-pases ms/fot (draws+disp):");
	for (int k = 1; k < Subs; ++k)
		if (g_state->id[k] != NativeRender::NoPass)
			std::printf(" %s %.3f (%.1f)", kSub[k], ms[k], g_sub_draws[k].exchange(0) / d);
	std::printf("\n");
	std::fflush(stdout);
}


// (luz) Con KYTY_POSTNATIVO=1..3 (y KYTY_TECHO=1) tambien se miden los ambitos de ILUMINACION del juego. SOLO MIDE.
// Ghidra (eboot 01.007.000), llamadores de FUN_00999220 (ScopeBegin) con el nombre literal (funcion -> ambito):
//   GI: 0x9a2c20 GIPreFrameAsync, 0x9a3720 GIPreFrame, 0x9a45e0 GIRender, 0x9a2ee0 GIPostFrame, 0x9ab420 GICaptureSkyProbe,
//       0x9bb670 GITreeManager::UpdateTree (bajo AllScenesUpdate 0xa16c60).
//   Luces: 0x9e3240 GPULightCulling, 0x9a15b0/0x9a1640 CreateClusters, 0x9dec40 RenderLights, 0x9e0a00 ResolveLighting,
//       0x9dffe0 ResolveSunShadows, 0xa98040 ParticleLighting.
//   Sombras: 0xa11220 CascadedShadows, 0xa10e20 SpotShadows, 0x9d1060/0x9d1fc0 AmbientShadowing.
//   IBL/SSR: 0x9d66d0 RenderIBL, 0x9d5eb0 IBLOcclusion, 0xa079f0 RenderSpecularIBL, 0x9d5120 RenderSSR, 0x9d6030 SSRClear /
//       SSRPreRender, 0x9d6110 SSRGetActiveTiles, 0x9d54c0 SSRCreateRays, 0x9d5880 SSRRaymarch.
//   Cielo/niebla: 0xa090e0 SkyboxOpaque / SkyboxAlpha (escena diferida), 0x9949d0 ParticleGroup_Fog.
// Por ambito (inclusivo, cualquier cola): ms/fot del hilo GPU (mapa, Cut::InclNs), draws, dispatches, pipelines distintos
// y huella igual/DIFERENTE al fotograma anterior. Linea "[luz]" cada 5 s junto a [postnativo]. Los que no aparecen: no salen.
namespace Luz {

constexpr int      N = 27;
inline const char* kName[N] = {"GIPreFrameAsync", "GIPreFrame",       "GIRender",          "GIPostFrame",      "GICaptureSkyProbe",
                               "GITreeManager::UpdateTree",           "GPULightCulling",   "CreateClusters",   "RenderLights",
                               "ResolveLighting", "ResolveSunShadows", "ParticleLighting", "CascadedShadows",  "SpotShadows",
                               "AmbientShadowing", "RenderIBL",       "IBLOcclusion",      "RenderSpecularIBL", "RenderSSR",
                               "SSRClear",        "SSRPreRender",     "SSRGetActiveTiles", "SSRCreateRays",    "SSRRaymarch",
                               "SkyboxOpaque",    "SkyboxAlpha",      "ParticleGroup_Fog"};

struct Scope {
	uint32_t              id = NativeRender::NoPass;
	uint32_t              draws = 0, disp = 0;
	uint64_t              hash = 0xcbf29ce484222325ull, prev = 0;
	bool                  have_prev = false;
	std::vector<uint64_t> pipes;
	// (luz-nativa, medida de fase 1) bytes PM4 del tramo: todo el paquete (b) y sin los que escriben memoria
	// (RELEASE_MEM / EVENT_WRITE_EOP / WRITE_DATA, con direcciones de anillo por fotograma) (bs); user data SH (ud).
	uint64_t bh = 0xcbf29ce484222325ull, bsh = 0xcbf29ce484222325ull, udh = 0xcbf29ce484222325ull;
	uint64_t bprev = 0, bsprev = 0, udprev = 0;
	uint32_t pk = 0, dw = 0, memw = 0, ud = 0;
	// acumulado de la ventana
	uint64_t w_draws = 0, w_disp = 0, w_pipes = 0, w_frames = 0, w_same = 0, w_diff = 0;
	uint64_t w_bsame = 0, w_bssame = 0, w_udsame = 0, w_pk = 0, w_dw = 0, w_memw = 0, w_ud = 0;
	int64_t  last_ns = 0;
};
struct State {
	Scope    s[N];
	bool     any = false;
	uint32_t tries = 0;
};
inline State*      g = nullptr;
inline std::mutex  g_mx; // (hilo GPU escribe la ventana; el informe la lee)

inline void Resolve(State& st) {
	if (!NativeRender::PassRegistry::Ready() || (++st.tries & 1023u) != 1u) return;
	for (auto& sc : st.s)
		if (sc.id == NativeRender::NoPass)
			for (uint32_t i = 0, n = NativeRender::PassRegistry::Count(); i < n; ++i)
				if (NativeRender::PassRegistry::SameName(NativeRender::PassRegistry::Name(i), kName[&sc - st.s])) {
					sc.id  = i;
					st.any = true;
					break;
				}
}

inline void Work(uint32_t opcode, bool draw, uint64_t vs, uint64_t ps, uint64_t cs, uint64_t cb0, uint64_t z) {
	if (!NativeRender::Cut::OnGpuThread()) return;
	if (g == nullptr) g = new State;
	Resolve(*g);
	if (!g->any) return;
	const uint64_t pipe = draw ? Fold(Fold(vs, ps), 1) : Fold(cs, 2);
	const uint64_t q    = static_cast<uint64_t>(NativeRender::Cut::Queue());
	for (auto& sc : g->s) {
		if (sc.id == NativeRender::NoPass || !NativeRender::Cut::IsOpen(sc.id)) continue;
		if (draw) ++sc.draws; else ++sc.disp;
		sc.hash = Fold(Fold(Fold(Fold(sc.hash, opcode), pipe), q), draw ? Fold(cb0, z) : 0);
		sc.pipes.push_back(pipe);
	}
}

// Cada paquete PM4 (hilo GPU) con su longitud ya validada: se pliega en los ambitos abiertos.
inline void Packet(const uint32_t* p, uint32_t len, uint32_t opcode, bool mem_write, bool sh_write) {
	if (g == nullptr || !g->any || !NativeRender::Cut::OnGpuThread()) return;
	const uint32_t n = len < 64u ? len : 64u;
	for (auto& sc : g->s) {
		if (sc.id == NativeRender::NoPass || !NativeRender::Cut::IsOpen(sc.id)) continue;
		++sc.pk, sc.dw += len;
		uint64_t h = sc.bh;
		for (uint32_t i = 0; i < n; ++i) h = Fold(h, p[i]);
		sc.bh = h;
		if (mem_write) {
			++sc.memw;
			sc.bsh = Fold(sc.bsh, opcode);
		} else {
			sc.bsh = h ^ sc.bsh * 31u;
		}
		if (sh_write) {
			++sc.ud;
			for (uint32_t i = 1; i < n; ++i) sc.udh = Fold(sc.udh, p[i]);
		}
	}
}

inline void Flip() {
	if (g == nullptr) return;
	std::lock_guard<std::mutex> lk(g_mx);
	for (auto& sc : g->s) {
		if (sc.draws + sc.disp != 0) {
			++sc.w_frames;
			if (sc.have_prev) {
				sc.w_bsame += sc.bh == sc.bprev, sc.w_bssame += sc.bsh == sc.bsprev, sc.w_udsame += sc.udh == sc.udprev;
			}
			sc.bprev = sc.bh, sc.bsprev = sc.bsh, sc.udprev = sc.udh;
			sc.w_pk += sc.pk, sc.w_dw += sc.dw, sc.w_memw += sc.memw, sc.w_ud += sc.ud;
			if (sc.have_prev) ++(sc.hash == sc.prev ? sc.w_same : sc.w_diff);
			sc.prev = sc.hash, sc.have_prev = true;
			sc.w_draws += sc.draws, sc.w_disp += sc.disp, sc.w_pipes += Distinct(sc.pipes);
		} else {
			sc.have_prev = false;
		}
		sc.draws = sc.disp = 0;
		sc.hash = 0xcbf29ce484222325ull;
		sc.bh = sc.bsh = sc.udh = 0xcbf29ce484222325ull;
		sc.pk = sc.dw = sc.memw = sc.ud = 0;
		sc.pipes.clear();
	}
}

inline void Window(double busy_ms, double frames) {
	if (Mode() == 0 || frames <= 0 || g == nullptr || !g->any) return;
	std::lock_guard<std::mutex> lk(g_mx);
	std::printf("[luz] ambitos de iluminacion, ms/fot hilo GPU inclusivo (render ocupado %.2f) | nombre ms (fot con trabajo, "
	            "draws disp pipelines /fot, huella igual%%):", busy_ms);
	for (auto& sc : g->s) {
		if (sc.id == NativeRender::NoPass) continue;
		const int64_t now = NativeRender::Cut::InclNs(sc.id);
		const double  ms  = static_cast<double>(now - sc.last_ns) * 1e-6 / frames;
		sc.last_ns        = now;
		const double d    = sc.w_frames > 0 ? static_cast<double>(sc.w_frames) : 1.0;
		const double sd   = static_cast<double>(sc.w_same + sc.w_diff);
		std::printf(" | %s %.3f (%llu, %.1f %.1f %.1f, %.0f%%)", kName[&sc - g->s], ms,
		            static_cast<unsigned long long>(sc.w_frames), sc.w_draws / d, sc.w_disp / d, sc.w_pipes / d,
		            sd > 0 ? 100.0 * sc.w_same / sd : 0.0);
		sc.w_draws = sc.w_disp = sc.w_pipes = sc.w_same = sc.w_diff = 0; // (w_frames: tras [luz-nativa])
	}
	std::printf("\n");
	// (luz-nativa) dato decisivo de la fase 1: si el PM4 del tramo (y su user data) es igual palabra a palabra al del
	// fotograma anterior, el tramo se podria reproducir sin traducir; si solo cambian los paquetes que escriben memoria,
	// esos se ejecutarian aparte. Sin descriptores en memoria (V#/T#/S#): eso falta aun.
	std::printf("[luz-nativa] ambito (paquetes dw escr.mem userdata /fot; PM4 igual%% | igual sin escr.mem%% | user data igual%%):");
	for (auto& sc : g->s) {
		if (sc.id == NativeRender::NoPass || sc.w_frames == 0) continue;
		const double d  = static_cast<double>(sc.w_frames);
		const double dp = sc.w_frames > 1 ? static_cast<double>(sc.w_frames - 1) : 1.0;
		std::printf(" | %s %.1f %.0f %.1f %.1f; %.0f%% %.0f%% %.0f%%", kName[&sc - g->s], sc.w_pk / d, sc.w_dw / d, sc.w_memw / d,
		            sc.w_ud / d, 100.0 * sc.w_bsame / dp, 100.0 * sc.w_bssame / dp, 100.0 * sc.w_udsame / dp);
		sc.w_bsame = sc.w_bssame = sc.w_udsame = sc.w_pk = sc.w_dw = sc.w_memw = sc.w_ud = 0;
	}
	for (auto& sc : g->s) sc.w_frames = 0;
	std::printf("\n");
	std::fflush(stdout);
}

} // namespace Luz

} // namespace PostNativo
