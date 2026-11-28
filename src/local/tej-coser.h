#pragma once
// (tej-repro, PASO A) Ademas: el trabajador captura por paquete la lista resuelta de draws y binds (tej-repro.h) y el
// render, en el punto de cose, la reproduce por el dispatcher de DESCARTE sobre una copia de su estado y la compara draw a
// draw con lo que grabo de verdad. Linea [tej] F2 reproduccion. Imagen intacta: nada se graba en un buffer real.
// (tej-coser) KYTY_TEJ=3, FASE 2 del tejido, ETAPA 1: SOLO MIDE (nada se ejecuta, nada se salta, imagen identica a KYTY_TEJ=2).
// Sin VK_KHR_maintenance7 (vulkanWindow.cpp, linea "[tej] F2 coser") KYTY_TEJ=3 es KYTY_TEJ=2 y esto no corre.
// Por cada paquete de 64 draws que el trabajador publica y el render alcanza (lado B de la ventana A/B):
//  - trabajador: estado de rendering de SU buffer al abrir el paquete (la herencia que tendria el secundario) y sonda de
//    llamadas (tej-coser-sonda.h) del paquete: hash + barreras/copias/dispatches/Begin/EndRendering dentro.
//    "valido como secundario" = empieza dentro de un rendering y dentro solo hay draws (ClearAttachments incluido).
//  - render: en el punto de cose, su rendering abierto frente a la herencia (CommandBuffer::RenderingWith: mismos
//    attachments, vistas, layouts, tamano y capas = compatibles) y la misma sonda sobre lo que el render graba por el
//    camino normal para ese paquete, hasta el primer draw del paquete siguiente (o el fin de la DCB).
//  - VERIFICACION: hash del trabajador == hash del render (misma secuencia de trabajo con los mismos escalares).
// Linea [tej] F2 coser: paquetes y draws que se coserian HOY con imagen demostrada por hash (F1d pasa + compatible +
// valido como secundario + sin huecos + hash igual), y cada filtro por separado.
// Por que no se graba todavia en secundarios: etapa 1 mide con la grabacion primaria actual del trabajador (que se tira);
// la secuencia de llamadas de un paquete que cumple los filtros es la misma que tendria en un secundario (no hay Begin/
// EndRendering ni barreras dentro), asi que el recuento es el de la etapa 2 sin tocar la segmentacion que usa F2p.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "local/tej-coser-sonda.h"
#include "local/tej-gen.h"
#include "local/tej-repro.h" // (tej-repro, paso A: captura + reproduccion en sombra)
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "local/vulkan-recording.h"
#endif
#include <chrono>
#include <memory>
#include <vector>

namespace TejCoser {

[[nodiscard]] inline bool On() noexcept { return Wanted() && g_m7.load(std::memory_order_relaxed); }

// Lo que el trabajador deja en cada paquete (lo escribe antes de publicar: Slot::count con release).
struct PkInfo {
	Libs::Graphics::RenderState rs;
	bool                        rs_on   = false; // el buffer del trabajador estaba dentro de un rendering al abrir
	bool                        probed  = false;
	uint64_t                    end_key = ~uint64_t {0}; // primer draw del paquete siguiente (~0: fin de la DCB)
	Probe                       probe;
	// (tej-repro) Lista resuelta de draws y binds del paquete (tej-repro.h) y los duenos que sus paquetes necesitan.
	std::vector<uint64_t>                    cap;
	std::vector<std::shared_ptr<const void>> own;
	bool                                     cap_on = false, cap_bad = false;
	bool                                     sampled = true; // (A4) paquete muestreado (1 de cada Div())
};
// (A4) KYTY_TEJ_REPRO_MUESTREO (4 por defecto): solo 1 de cada N paquetes se captura, se sigue y se compara (el coste
// del seguidor/sonda/comparacion en el render baja a 1/N); las cifras por fotograma se escalan x N al imprimir.
[[nodiscard]] inline uint32_t Div() noexcept {
	static const uint32_t d = [] {
		const char* v = std::getenv("KYTY_TEJ_REPRO_MUESTREO");
		const int   n = v != nullptr ? std::atoi(v) : 4;
		return static_cast<uint32_t>(n < 1 ? 1 : n > 64 ? 64 : n);
	}();
	return d;
}

inline void WorkerOpen(PkInfo& p) {
	p.rs_on = false, p.probed = false, p.end_key = ~uint64_t {0};
	p.cap_on = false;
	if (TejRepro::t_cap != nullptr) TejRepro::t_cap = nullptr, TejRepro::t_own = nullptr, TejRepro::Sync();
	if (!On()) return;
	static thread_local uint32_t seq = 0;
	p.sampled          = (seq++ % Div()) == 0;
	TejRepro::t_note   = true; // (A4) los bloques subidos en paquetes no muestreados se reutilizan en los muestreados
	if (!p.sampled) return;
	auto* rec = Libs::Graphics::CommandScheduler::ThreadRecorder();
	if (rec != nullptr && rec->command && !rec->command->IsInvalid() && rec->command->IsRendering()) {
		p.rs    = rec->command->CurrentRenderState();
		p.rs_on = true;
	}
	Start();
	if (t_on) {
		// (paso A2) el camino directo del trabajador vuelve a grabar pipeline/estado dinamico/IB al abrir: la lista no
		// depende de lo que SU buffer ya tenia (en el paso B se reproduce sobre el buffer del render). Solo binds de mas.
		if (rec != nullptr && rec->command && !rec->command->IsInvalid()) {
			const uint64_t g0 = rec->command->GraphicsGeneration();
			rec->command->InvalidateGraphicsState();
			TejGen::Note(g0, rec->command->GraphicsGeneration()); // (tej-huecos: el IB ya ligado se reutiliza sin buscarlo)
		}
		// (tej-repro) captura de este paquete (la escriben los vkCmd* del trabajador y sus paquetes de draw/descriptores)
		p.cap.clear(), p.own.clear();
		TejRepro::t_cap = &p.cap, TejRepro::t_own = &p.own, TejRepro::t_bad = false;
		TejRepro::Sync();
	}
}
inline void WorkerClose(PkInfo& p) {
	if (!t_on) return;
	p.probe  = Stop();
	p.probed = true;
	if (TejRepro::t_cap == &p.cap) {
		p.cap_bad = TejRepro::t_bad, p.cap_on = true;
		TejRepro::t_cap = nullptr, TejRepro::t_own = nullptr;
		TejRepro::Sync();
	}
}

enum K : uint32_t {
	KPk, KDraws, KF1d, KNoHoles, KWValid, KWInRendering, KCompat, KHashEq, KHashEqSameDraws, KRValid, KAll, KAllDraws,
	KAbandon, KWBarrier, KWCopy, KWDispatch, KWBegin, KWEnd, KDiffDraws, KDiffArgs, KNoProbe, KDcbEnd,
	KDSeq, KDIds, KDClr, KDArg0, KDArg1, KDArg2, KDArg3, KDArg4, KDRel0, KDRel1, KDRel2, KDRel3, KDRel4, KDOnlyShift,
	KDNone,
	// (tej-repro) reproduccion en sombra
	RPk, RFunnel, RFunnelDraws, RNoCap, RBad, RFail, RFirst, ROverflow, RDrawsDiff, RArgs, RArgsNh,
	RCat0, RCat1, RCat2, RCat3, RCat4, RNh0, RNh1, RNh2, RNh3, RNh4, REq, REqDraws, REqNh, REqNhDraws, ROk, ROkDraws,
	ROkNh, ROkNhDraws, RShadowNs, RShadowDraws, RWinNs, RWinDraws, RWords,
	RPushHandleOnly, RDrawsMore, RDrawsLess, RDrawsDcbEnd, RPushQ0, RPushQEnd = RPushQ0 + TejRepro::kPushQ,
	// (A4) comparacion ALINEADA por ID de draw (por draw salvo los AK*, por paquete)
	AP, AXr, AXrHole, AXrSub, AXrIn, AXrOut, AXw, AXwSub, AXwOther, AEq, AEqNh, AArgNh, ANh0, ANh1, ANh2, ANh3, ANh4,
	APw0, APwEnd = APw0 + TejRepro::kPushQ, AKClean, AKOkNh, AKOkNhDraws, AKOkNhF, AKOkNhFDraws, AKPairsOk, AKPairsOkDraws,
	AKOnlyHoles, CSkip, COpenTicks, CFinTicks, CCalls, KN
};
inline std::atomic<uint64_t> g_k[KN] {};
inline void                  Add(K k, uint64_t v = 1) { g_k[k].fetch_add(v, std::memory_order_relaxed); }

// Render (solo el hilo de render).
struct Live {
	bool     on = false;
	uint64_t end_key = 0;
	uint32_t draws = 0;
	bool     f1d = false, no_holes = false, compat = false, w_valid = false;
	Probe    w;
	// (tej-repro) resultado de la reproduccion en sombra (hecha al abrir) y comienzo de la ventana del render
	bool                                  r_on = false, r_ok = false, r_bad = false, r_first = false;
	TejRepro::Acc                         shadow;
	uint64_t                              shadow_ns = 0;
	std::chrono::steady_clock::time_point t0;
};
inline Live g_live;
inline bool g_opened = false; // (A4) el render abrio algun paquete en esta DCB
// (tej-repro) Seguidor de estado del render: desde el primer paquete medido de la DCB hasta su fin.
inline TejRepro::Acc g_track;
inline bool          g_track_on = false;
inline std::vector<TejRepro::DrawRec> g_rdl, g_sdl; // (A4) draws del render y de la sombra con su ID
inline std::vector<uint64_t>          g_holes;
inline uint64_t                       g_calls0 = 0; // (A4) TejRepro::t_done al empezar la ventana del render

// (A4) Compara por ID (clave PM4, sub), no por posicion. Devuelve si todas las parejas son iguales sin handles.
inline void AlignCompare(bool funnel) {
	using TejRepro::DrawRec;
	auto&      r  = g_rdl;
	auto&      w  = g_sdl;
	const auto lt = [](const DrawRec& a, const DrawRec& b) { return a.key != b.key ? a.key < b.key : a.sub < b.sub; };
	if (!std::is_sorted(r.begin(), r.end(), lt)) std::sort(r.begin(), r.end(), lt);
	if (!std::is_sorted(w.begin(), w.end(), lt)) std::sort(w.begin(), w.end(), lt);
	std::sort(g_holes.begin(), g_holes.end());
	const uint64_t wlo = w.empty() ? 0 : w.front().key, whi = w.empty() ? 0 : w.back().key;
	const auto     wkey = [&](uint64_t k) {
		return std::binary_search(w.begin(), w.end(), DrawRec {k, 0}, [](const DrawRec& a, const DrawRec& b) { return a.key < b.key; });
	};
	const auto rkey = [&](uint64_t k) {
		return std::binary_search(r.begin(), r.end(), DrawRec {k, 0}, [](const DrawRec& a, const DrawRec& b) { return a.key < b.key; });
	};
	uint64_t pairs = 0, ok = 0, xr = 0, xr_hole = 0, xw = 0;
	size_t   i = 0, j = 0;
	while (i < r.size() || j < w.size()) {
		if (j == w.size() || (i < r.size() && lt(r[i], w[j]))) { // draw del render sin pareja
			const auto& a = r[i++];
			++xr;
			if (std::binary_search(g_holes.begin(), g_holes.end(), a.key)) Add(AXrHole), ++xr_hole;
			else if (a.sub > 0 && wkey(a.key)) Add(AXrSub);
			else if (a.key >= wlo && a.key <= whi) Add(AXrIn);
			else Add(AXrOut);
			continue;
		}
		if (i == r.size() || lt(w[j], r[i])) { // draw del trabajador sin pareja
			const auto& b = w[j++];
			++xw;
			Add(b.sub > 0 && rkey(b.key) ? AXwSub : AXwOther);
			continue;
		}
		const auto& a = r[i++];
		const auto& b = w[j++];
		++pairs;
		bool eq = a.v == b.v, eqn = a.vn == b.vn;
		if (!eqn) Add(AArgNh);
		for (uint32_t c = 0; c < TejRepro::kState; ++c) {
			if (a.x[c] != b.x[c]) eq = false;
			if (a.xn[c] != b.xn[c]) eqn = false, Add(static_cast<K>(ANh0 + c));
		}
		if (a.xn[TejRepro::CPush] != b.xn[TejRepro::CPush])
			for (uint32_t q = 0; q < TejRepro::kPushQ; ++q)
				if (a.pw[q] != b.pw[q]) Add(static_cast<K>(APw0 + q));
		if (eq) Add(AEq);
		if (eqn) Add(AEqNh), ++ok;
	}
	Add(AP, pairs), Add(AXr, xr), Add(AXw, xw);
	const bool pairs_ok = pairs != 0 && ok == pairs;
	if (xr == 0 && xw == 0) Add(AKClean);
	if (pairs_ok) Add(AKPairsOk), Add(AKPairsOkDraws, pairs);
	if (pairs_ok && xw == 0 && xr == xr_hole && xr != 0) Add(AKOnlyHoles); // (todo igual salvo los huecos: el render los dibuja)
	if (pairs_ok && xr == 0 && xw == 0) {
		Add(AKOkNh), Add(AKOkNhDraws, pairs);
		if (funnel) Add(AKOkNhF), Add(AKOkNhFDraws, pairs);
	}
}

inline void ReproFinish(Live& l, bool dcb_end) {
	if (!l.r_on) return;
	l.r_on = false;
	const auto& r  = g_track;
	const auto& sh = l.shadow;
	const uint64_t win = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - l.t0).count());
	Add(RPk);
	const bool funnel = l.f1d && l.no_holes && l.w_valid && l.compat;
	if (funnel) Add(RFunnel), Add(RFunnelDraws, l.draws);
	if (l.r_first) Add(RFirst);
	if (l.r_bad) Add(RBad);
	if (!l.r_ok) {
		Add(RFail);
		return;
	}
	if (r.overflow || sh.overflow) Add(ROverflow);
	Add(RShadowNs, l.shadow_ns), Add(RShadowDraws, sh.draws);
	Add(RWinNs, win), Add(RWinDraws, r.draws);
	bool eq = sh.draws == r.draws && !r.overflow && !sh.overflow, eqn = eq;
	if (sh.draws != r.draws) {
		// (paso A3, diag) limite del paquete: el render conto de mas (no vio el primer draw del siguiente) o de menos
		Add(RDrawsDiff), Add(r.draws > sh.draws ? RDrawsMore : RDrawsLess);
		if (dcb_end) Add(RDrawsDcbEnd);
	}
	if (sh.hd != r.hd) Add(RArgs), eq = false;
	if (sh.hdn != r.hdn) Add(RArgsNh), eqn = false;
	for (uint32_t c = 0; c < TejRepro::kState; ++c) {
		if (sh.h[c] != r.h[c]) Add(static_cast<K>(RCat0 + c)), eq = false;
		if (sh.hn[c] != r.hn[c]) Add(static_cast<K>(RNh0 + c)), eqn = false;
	}
	// (paso A2, diag push) distinto solo por el layout (handle) o por el contenido; y que palabra de 8 bytes difiere.
	if (sh.h[TejRepro::CPush] != r.h[TejRepro::CPush] && sh.hn[TejRepro::CPush] == r.hn[TejRepro::CPush]) Add(RPushHandleOnly);
	if (sh.hn[TejRepro::CPush] != r.hn[TejRepro::CPush])
		for (uint32_t i = 0; i < TejRepro::kPushQ; ++i)
			if (sh.pq[i] != r.pq[i]) Add(static_cast<K>(RPushQ0 + i));
	if (l.r_bad) eq = eqn = false;
	if (eq) Add(REq), Add(REqDraws, r.draws);
	if (eqn) Add(REqNh), Add(REqNhDraws, r.draws);
	if (eq && funnel) Add(ROk), Add(ROkDraws, r.draws);
	if (eqn && funnel) Add(ROkNh), Add(ROkNhDraws, r.draws);
	Add(CCalls, TejRepro::t_done - g_calls0);
	const uint64_t c0 = __rdtsc();
	AlignCompare(funnel); // (A4)
	Add(CFinTicks, __rdtsc() - c0);
}
// Al abrir: la lista del trabajador se reproduce YA, por el dispatcher de descarte, sobre una copia del estado del render
// (lo que el paso B haria en su buffer real); el render sigue por su camino normal y Finish compara draw a draw.
inline void ReproOpen(const PkInfo& p, Libs::Graphics::RenderContext& rc) {
	auto& l = g_live;
	l.r_on  = false;
	if (!p.cap_on) {
		Add(RNoCap);
		return;
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	l.r_first = !g_track_on;
	if (!g_track_on) {
		g_track.dl = &g_rdl;
		if (g_rdl.capacity() < 256) g_rdl.reserve(256), g_sdl.reserve(256), g_holes.reserve(64);
		g_track.ResetAll(), g_track_on = true;
		// (paso A2) el seguidor empieza vacio: el camino directo del render vuelve a grabar su estado (lo que ya tenia
		// su buffer no se veria). Solo binds de mas, una vez por DCB y solo en la sombra.
		auto& sch = rc.GetCommandScheduler();
		if (Libs::Graphics::CommandScheduler::ThreadRecorder() == nullptr && sch.Active() && !sch.Current().IsInvalid())
			sch.Current().InvalidateGraphicsState();
	}
	l.shadow    = g_track;
	l.shadow.dl = &g_sdl; // (A4)
	l.shadow.ResetDraws();
	g_holes.clear();
	const uint64_t key0 = TejRepro::t_key;
	const uint32_t sub0 = TejRepro::t_sub;
	TejRepro::t_holes   = &g_holes;
	l.r_bad = p.cap_bad;
	Add(RWords, p.cap.size());
	const bool saved_bad = TejRepro::t_bad;
	TejRepro::t_bad      = false; // (paso A2) antes: el t_bad del render (nunca se limpiaba) marcaba el 100 % como no capturable
	TejRepro::t_acc = &l.shadow, TejRepro::t_cap = nullptr, TejRepro::Sync();
	const auto t = std::chrono::steady_clock::now();
	l.r_ok = LocalVulkanRecording::TejReplay(p.cap.data(), p.cap.size(), nullptr, LocalVulkanRecording::TejStubDispatch());
	l.shadow_ns = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count());
	if (TejRepro::t_bad) l.r_bad = true;
	TejRepro::t_bad = saved_bad;
	TejRepro::t_key = key0, TejRepro::t_sub = sub0, TejRepro::t_holes = nullptr; // (A4)
	g_calls0        = TejRepro::t_done;
	g_track.ResetDraws();
	TejRepro::t_acc = &g_track, TejRepro::Sync();
	l.r_on = true;
	l.t0   = std::chrono::steady_clock::now();
#endif
}
inline void ReproDcbEnd() {
	g_track_on = false;
	if (TejRepro::t_acc == &g_track) TejRepro::t_acc = nullptr, TejRepro::Sync();
}

inline void Finish(bool dcb_end) {
	auto& l = g_live;
	if (!l.on) return;
	l.on            = false;
	ReproFinish(l, dcb_end); // (tej-repro)
	if (Div() > 1 && !dcb_end) ReproDcbEnd(); // (A4) muestreo: el seguidor se apaga hasta el siguiente paquete muestreado
	const Probe rp  = Stop();
	const bool  eq  = rp.hash == l.w.hash;
	Add(KPk);
	Add(KDraws, l.draws);
	if (dcb_end) Add(KDcbEnd);
	if (l.f1d) Add(KF1d);
	if (l.no_holes) Add(KNoHoles);
	if (l.w_valid) Add(KWValid);
	if (l.compat) Add(KCompat);
	if (rp.SecondaryOk()) Add(KRValid);
	if (eq) Add(KHashEq);
	else if (rp.draws != l.w.draws) Add(KDiffDraws);
	else {
		// (diag) Que difiere con los mismos draws. Un paquete puede sumar en varias columnas.
		Add(KDiffArgs);
		const auto& w = l.w;
		bool any = false, real = false;
		if (rp.seq != w.seq) Add(KDSeq), any = real = true;
		if (rp.ids != w.ids) Add(KDIds), any = real = true;
		if (rp.clr != w.clr) Add(KDClr), any = real = true;
		for (uint32_t k = 0; k < Probe::kArgs; ++k) {
			if (rp.arg[k] == w.arg[k]) continue;
			any = true;
			Add(static_cast<K>(KDArg0 + k));
			if (rp.rel[k] != w.rel[k]) Add(static_cast<K>(KDRel0 + k)), real = true;
		}
		if (any && !real) Add(KDOnlyShift);
		if (!any) Add(KDNone);
	}
	if (l.w.barriers != 0) Add(KWBarrier);
	if (l.w.copies != 0) Add(KWCopy);
	if (l.w.dispatches != 0) Add(KWDispatch);
	if (l.w.begins != 0) Add(KWBegin);
	if (l.w.ends != 0) Add(KWEnd);
	if (eq && l.f1d && l.no_holes && l.w_valid && l.compat) Add(KAll), Add(KAllDraws, l.draws);
}
// Primero en cada GpuDraw: el paquete medido acaba en el primer draw del siguiente.
inline void RenderTick(uint64_t key) {
	if (g_live.on && key >= g_live.end_key) Finish(false);
}
// En el punto de cose (el render alcanza el primer draw del paquete publicado).
inline void RenderOpen(const PkInfo& p, uint32_t draws, bool f1d, bool no_holes, Libs::Graphics::RenderContext& r) {
	if (g_live.on) Finish(false), Add(KAbandon);
	if (!On()) return;
	TejRepro::t_note = true; // (A4) el render apunta sus subidas en ventanas B aunque no siga (bloques reutilizados)
	g_opened        = true;
	if (!p.sampled) {
		Add(CSkip);
		return;
	}
	const uint64_t o0 = __rdtsc();
	struct OT {
		uint64_t t;
		~OT() { Add(COpenTicks, __rdtsc() - t); }
	} ot {o0};
	if (!p.probed) {
		Add(KNoProbe);
		return;
	}
	auto&                          sch = r.GetCommandScheduler();
	const Libs::Graphics::CommandBuffer* cb =
	    Libs::Graphics::CommandScheduler::ThreadRecorder() == nullptr && sch.Active() ? &sch.Current() : nullptr;
	auto& l    = g_live;
	l.on       = true;
	l.end_key  = p.end_key;
	l.draws    = draws;
	l.f1d      = f1d;
	l.no_holes = no_holes;
	l.w        = p.probe;
	l.w_valid  = p.rs_on && p.probe.SecondaryOk();
	if (p.rs_on) Add(KWInRendering);
	l.compat = p.rs_on && cb != nullptr && cb->RenderingWith(p.rs);
	ReproOpen(p, r); // (tej-repro: antes de Start, la sombra no entra en la sonda)
	Start();
}
inline void RenderDcbEnd() {
	if (!g_opened) TejRepro::t_note = false; // (A4) DCB sin ningun paquete abierto (ventana A): deja de apuntar
	g_opened = false;
	if (g_live.on) {
		if (g_live.end_key == ~uint64_t {0}) Finish(true);
		else g_live.r_on = false, (void)Stop(), g_live.on = false, Add(KAbandon);
	}
	ReproDcbEnd(); // (tej-repro)
}

inline void Print(double frames) {
	if (!Wanted() || frames <= 0) return;
	uint64_t v[KN];
	for (uint32_t i = 0; i < KN; ++i) v[i] = g_k[i].exchange(0, std::memory_order_relaxed);
	const double f  = frames / Div(); // (A4) muestreo 1/N: cifras por fotograma escaladas x N
	const double pk = static_cast<double>(v[KPk]);
	const auto   pc = [&](uint64_t x) { return pk > 0 ? 100.0 * static_cast<double>(x) / pk : 0.0; };
	if (!On()) {
		std::printf("[tej] F2 coser: APAGADO (sin VK_KHR_maintenance7: KYTY_TEJ=3 = KYTY_TEJ=2)\n");
		return;
	}
	if (!g_installed.load(std::memory_order_relaxed)) {
		std::printf("[tej] F2 coser: sin sonda (compilacion sin KYTY_LOCAL_VULKAN_RECORDING): no se puede verificar\n");
		return;
	}
	std::printf("[tej] F2 coser (etapa 1, solo mide) | paquetes medidos %.1f/fot (%.0f draws/fot; fin de DCB %.1f, abandonados %.1f,"
	            " sin sonda %.1f) | SE COSERIAN CON HASH IGUAL %.1f pk/fot = %.0f draws/fot (%.1f %% de los medidos) | filtros: F1d pasa"
	            " %.1f %%, sin huecos %.1f %%, trabajador dentro de rendering %.1f %%, valido como secundario %.1f %%, rendering del"
	            " render compatible %.1f %%, render sin barreras/copias dentro %.1f %%, hash igual %.1f %% | trabajador con dentro:"
	            " barreras %.1f, copias %.1f, dispatches %.1f, BeginRendering %.1f, EndRendering %.1f pk/fot | verificacion: distintos"
	            " %.1f pk/fot (otra cuenta de draws %.1f, mismos draws otros argumentos %.1f)\n",
	            pk / f, v[KDraws] / f, v[KDcbEnd] / f, v[KAbandon] / f, v[KNoProbe] / f, v[KAll] / f, v[KAllDraws] / f,
	            pc(v[KAll]), pc(v[KF1d]), pc(v[KNoHoles]), pc(v[KWInRendering]), pc(v[KWValid]), pc(v[KCompat]), pc(v[KRValid]),
	            pc(v[KHashEq]), v[KWBarrier] / f, v[KWCopy] / f, v[KWDispatch] / f, v[KWBegin] / f, v[KWEnd] / f,
	            (v[KDiffDraws] + v[KDiffArgs]) / f, v[KDiffDraws] / f, v[KDiffArgs] / f);
	// (diag) Mismos draws otros argumentos, por escalar: arg = difiere el valor; rel = difiere tambien relativo al primer
	// draw del paquete (no es un desplazamiento constante). SOLO DESPLAZAMIENTO = todo lo distinto es un offset constante
	// en firstIndex/vertexOffset/firstInstance (anillo propio del trabajador: equivalente al reproducir con remapeo).
	std::printf("[tej] F2 coser diag (mismos draws otros argumentos %.1f pk/fot) | llamadas no-draw distintas (barreras/copias/"
	            "Begin/End o sus cuentas) %.1f | orden/tipo de draws %.1f | ClearAttachments %.1f | indexCount %.1f (rel %.1f),"
	            " instanceCount %.1f (rel %.1f), firstIndex %.1f (rel %.1f), vertexOffset %.1f (rel %.1f), firstInstance %.1f"
	            " (rel %.1f) pk/fot | SOLO DESPLAZAMIENTO CONSTANTE %.1f pk/fot | sin causa vista %.1f\n",
	            v[KDiffArgs] / f, v[KDSeq] / f, v[KDIds] / f, v[KDClr] / f, v[KDArg0] / f, v[KDRel0] / f, v[KDArg1] / f,
	            v[KDRel1] / f, v[KDArg2] / f, v[KDRel2] / f, v[KDArg3] / f, v[KDRel3] / f, v[KDArg4] / f, v[KDRel4] / f,
	            v[KDOnlyShift] / f, v[KDNone] / f);
	// (tej-repro, paso A) Reproduccion en sombra: la lista del trabajador por el dispatcher de descarte sobre el estado del
	// render, comparada draw a draw con lo que el render grabo de verdad (estado vigente en cada draw por categoria).
	// traducir = ventana del render por draw menos el seguidor (~= lo que cuesta la sombra, mismo hasheo): estimacion.
	const double rpk    = static_cast<double>(v[RPk]);
	const auto   rpc    = [&](uint64_t x) { return rpk > 0 ? 100.0 * static_cast<double>(x) / rpk : 0.0; };
	const double rep_us = v[RShadowDraws] ? static_cast<double>(v[RShadowNs]) / 1000.0 / static_cast<double>(v[RShadowDraws]) : 0.0;
	const double win_us = v[RWinDraws] ? static_cast<double>(v[RWinNs]) / 1000.0 / static_cast<double>(v[RWinDraws]) : 0.0;
	const double tra_us = win_us > rep_us ? win_us - rep_us : 0.0;
	const double gain   = tra_us > rep_us ? static_cast<double>(v[ROkDraws]) / f * (tra_us - rep_us) / 1000.0 : 0.0;
	const double gainnh = tra_us > rep_us ? static_cast<double>(v[ROkNhDraws]) / f * (tra_us - rep_us) / 1000.0 : 0.0;
	std::printf("[tej] F2 reproduccion (paso A, sombra: nada llega al buffer real) | paquetes con lista %.1f/fot (sin lista %.1f,"
	            " primer paquete de la DCB %.1f, llamada no capturable %.1f, lista rota %.1f, ranuras llenas %.1f) | pasan embudo %.1f"
	            " pk/fot = %.0f draws/fot | REPRODUCIBLES exactos (embudo + estado igual en cada draw) %.1f pk/fot = %.0f draws/fot;"
	            " REPRODUCIBLES POR CONTENIDO (anillos de subida comparados por lo subido) %.1f pk/fot = %.0f draws/fot | iguales"
	            " %.1f %% (por contenido %.1f %%) | distintos por motivo (%% de paquetes):"
	            " cuenta de draws %.1f, args de draw %.1f (sin handles %.1f), pipeline %.1f (%.1f), sets %.1f (%.1f), push %.1f (%.1f),"
	            " VB/IB %.1f (%.1f), estado dinamico %.1f (%.1f) | coste: reproducir %.3f us/draw, traducir ~%.3f us/draw (ventana"
	            " %.3f) | lista %.0f palabras/pk | AHORRO ESTIMADO paso B %.2f ms/fot (por contenido %.2f)\n",
	            rpk / f, v[RNoCap] / f, v[RFirst] / f, v[RBad] / f, v[RFail] / f, v[ROverflow] / f, v[RFunnel] / f,
	            v[RFunnelDraws] / f, v[ROk] / f, v[ROkDraws] / f, v[ROkNh] / f, v[ROkNhDraws] / f, rpc(v[REq]), rpc(v[REqNh]),
	            rpc(v[RDrawsDiff]), rpc(v[RArgs]), rpc(v[RArgsNh]), rpc(v[RCat0]), rpc(v[RNh0]), rpc(v[RCat1]), rpc(v[RNh1]),
	            rpc(v[RCat2]), rpc(v[RNh2]), rpc(v[RCat3]), rpc(v[RNh3]), rpc(v[RCat4]), rpc(v[RNh4]), rep_us, tra_us, win_us,
	            rpk > 0 ? static_cast<double>(v[RWords]) / rpk : 0.0, gain, gainnh);
	{ // (A4) comparacion ALINEADA por ID de draw (clave PM4 + sub) y coste del render en B
		static uint64_t lt = 0;
		static auto     lc = std::chrono::steady_clock::now();
		const uint64_t  t  = __rdtsc();
		const auto      c  = std::chrono::steady_clock::now();
		const double    npt = lt != 0 && t > lt ? std::chrono::duration<double, std::nano>(c - lc).count() / static_cast<double>(t - lt) : 0.25;
		lt = t, lc = c;
		const double ap = static_cast<double>(v[AP]);
		const auto   dp = [&](uint64_t x) { return ap > 0 ? 100.0 * static_cast<double>(x) / ap : 0.0; };
		const double xr = static_cast<double>(v[AXr]), xw = static_cast<double>(v[AXw]);
		const auto   rp = [&](uint64_t x) { return xr > 0 ? 100.0 * static_cast<double>(x) / xr : 0.0; };
		const auto   wp = [&](uint64_t x) { return xw > 0 ? 100.0 * static_cast<double>(x) / xw : 0.0; };
		char   q[256] = "";
		size_t qa     = 0;
		for (uint32_t i = 0; i < TejRepro::kPushQ; ++i) {
			const uint64_t n = v[APw0 + i];
			if (n == 0 || qa + 24 >= sizeof(q)) continue;
			const int wr = std::snprintf(q + qa, sizeof(q) - qa, "%sdw%u-%u %.1f%%", qa ? " " : "", 2 * i, 2 * i + 1, dp(n));
			if (wr > 0) qa += static_cast<size_t>(wr);
		}
		const double gain_al = tra_us > rep_us ? static_cast<double>(v[AKOkNhFDraws]) / f * (tra_us - rep_us) / 1000.0 : 0.0;
		std::printf("[tej] F2 alineado (por ID = clave PM4 + sub, muestreo 1/%u, /fot escalado) | parejas %.0f draws/fot, extra render"
		            " %.0f (huecos del trabajador %.1f %%, vkCmdDraw de mas del mismo draw PM4 %.1f %%, clave dentro del paquete sin"
		            " pareja %.1f %%, fuera del rango del trabajador %.1f %%), extra trabajador %.0f (sub de mas %.1f %%, otros %.1f %%)"
		            " | parejas iguales exactas %.1f %%, por contenido %.1f %% | parejas distintas por (%% de parejas): args %.1f,"
		            " pipeline %.1f, sets %.1f, push %.1f, VB/IB %.1f, dinamico %.1f | push por palabra [%s] | paquetes: sin extras %.1f %%,"
		            " todas las parejas iguales %.1f %% (%.0f draws/fot), iguales salvo huecos %.1f %%, REPRODUCIBLES ALINEADOS %.1f pk/fot ="
		            " %.0f draws/fot (con embudo %.1f pk/fot = %.0f draws/fot) | AHORRO ESTIMADO paso B %.2f ms/fot | coste render B:"
		            " abrir+sombra %.3f ms/fot, comparar %.3f ms/fot, %.0f vkCmd seguidas/fot, paquetes no muestreados %.1f/fot\n",
		            Div(), ap / f, xr / f, rp(v[AXrHole]), rp(v[AXrSub]), rp(v[AXrIn]), rp(v[AXrOut]), xw / f, wp(v[AXwSub]),
		            wp(v[AXwOther]), dp(v[AEq]), dp(v[AEqNh]), dp(v[AArgNh]), dp(v[ANh0]), dp(v[ANh1]), dp(v[ANh2]), dp(v[ANh3]),
		            dp(v[ANh4]), qa ? q : "-", rpc(v[AKClean]), rpc(v[AKPairsOk]), v[AKPairsOkDraws] / f, rpc(v[AKOnlyHoles]),
		            v[AKOkNh] / f, v[AKOkNhDraws] / f, v[AKOkNhF] / f, v[AKOkNhFDraws] / f, gain_al,
		            static_cast<double>(v[COpenTicks]) * npt / 1e6 / frames, static_cast<double>(v[CFinTicks]) * npt / 1e6 / frames,
		            v[CCalls] / static_cast<double>(frames), v[CSkip] / static_cast<double>(frames));
	}
	// (paso A2) por que no es capturable (por nombre, llamadas/fot del trabajador) y de que es distinto el push.
	char   names[640] = "";
	size_t at         = 0;
	for (auto& e : TejRepro::g_bad) {
		const char* n = e.name.load(std::memory_order_relaxed);
		const uint64_t c = e.n.exchange(0, std::memory_order_relaxed);
		if (n == nullptr || c == 0 || at + 96 >= sizeof(names)) continue;
		const int w = std::snprintf(names + at, sizeof(names) - at, "%s%s %.1f", at ? ", " : "", n, c / f);
		if (w > 0) at += static_cast<size_t>(w);
	}
	const uint64_t other = TejRepro::g_bad_other.exchange(0, std::memory_order_relaxed);
	const uint64_t comp  = TejRepro::g_compute.exchange(0, std::memory_order_relaxed);
	char   q[256] = "";
	size_t qa     = 0;
	for (uint32_t i = 0; i < TejRepro::kPushQ; ++i) {
		const uint64_t c = v[RPushQ0 + i];
		if (c == 0 || qa + 24 >= sizeof(q)) continue;
		const int w = std::snprintf(q + qa, sizeof(q) - qa, "%sdw%u-%u %.0f%%", qa ? " " : "", 2 * i, 2 * i + 1, rpc(c));
		if (w > 0) qa += static_cast<size_t>(w);
	}
	std::printf("[tej] F2 reproduccion (2) | no capturable por nombre (/fot): %s%s (otros %.1f) | compute fuera del estado de"
	            " draw %.0f binds/fot | push (%% de paquetes): igual %.1f, valor distinto con contenido apuntado igual %.1f,"
	            " contenido distinto %.1f [%s] | cuenta de draws distinta: render mas %.1f, menos %.1f, al fin de la DCB %.1f"
	            " (%% de paquetes) | sustituidas por contenido: direcciones de push %.0f/fot, VB/IB %.0f/fot; subidas"
	            " apuntadas %.0f/fot (>64 KiB sin apuntar %.1f) | ranuras llenas %.1f %%\n",
	            at ? names : "ninguna", "", other / f, comp / f, 100.0 - rpc(v[RCat2]), rpc(v[RPushHandleOnly]),
	            rpc(v[RNh2]), qa ? q : "-", rpc(v[RDrawsMore]), rpc(v[RDrawsLess]), rpc(v[RDrawsDcbEnd]),
	            TejRepro::g_push_subst.exchange(0, std::memory_order_relaxed) / f,
	            TejRepro::g_vb_subst.exchange(0, std::memory_order_relaxed) / f,
	            TejRepro::g_content_n.exchange(0, std::memory_order_relaxed) / f,
	            TejRepro::g_content_big.exchange(0, std::memory_order_relaxed) / f, rpc(v[ROverflow]));
}

} // namespace TejCoser
