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
#include <atomic>
#include <cstdint>
#include <cstdio>

#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "local/tej-coser-sonda.h"
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
};

inline void WorkerOpen(PkInfo& p) {
	p.rs_on = false, p.probed = false, p.end_key = ~uint64_t {0};
	p.cap_on = false;
	if (TejRepro::t_cap != nullptr) TejRepro::t_cap = nullptr, TejRepro::t_own = nullptr, TejRepro::Sync();
	if (!On()) return;
	auto* rec = Libs::Graphics::CommandScheduler::ThreadRecorder();
	if (rec != nullptr && rec->command && !rec->command->IsInvalid() && rec->command->IsRendering()) {
		p.rs    = rec->command->CurrentRenderState();
		p.rs_on = true;
	}
	Start();
	if (t_on) {
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
	ROkNh, ROkNhDraws, RShadowNs, RShadowDraws, RWinNs, RWinDraws, RWords, KN
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
// (tej-repro) Seguidor de estado del render: desde el primer paquete medido de la DCB hasta su fin.
inline TejRepro::Acc g_track;
inline bool          g_track_on = false;

inline void ReproFinish(Live& l) {
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
	if (sh.draws != r.draws) Add(RDrawsDiff);
	if (sh.hd != r.hd) Add(RArgs), eq = false;
	if (sh.hdn != r.hdn) Add(RArgsNh), eqn = false;
	for (uint32_t c = 0; c < TejRepro::kState; ++c) {
		if (sh.h[c] != r.h[c]) Add(static_cast<K>(RCat0 + c)), eq = false;
		if (sh.hn[c] != r.hn[c]) Add(static_cast<K>(RNh0 + c)), eqn = false;
	}
	if (l.r_bad) eq = eqn = false;
	if (eq) Add(REq), Add(REqDraws, r.draws);
	if (eqn) Add(REqNh), Add(REqNhDraws, r.draws);
	if (eq && funnel) Add(ROk), Add(ROkDraws, r.draws);
	if (eqn && funnel) Add(ROkNh), Add(ROkNhDraws, r.draws);
}
// Al abrir: la lista del trabajador se reproduce YA, por el dispatcher de descarte, sobre una copia del estado del render
// (lo que el paso B haria en su buffer real); el render sigue por su camino normal y Finish compara draw a draw.
inline void ReproOpen(const PkInfo& p) {
	auto& l = g_live;
	l.r_on  = false;
	if (!p.cap_on) {
		Add(RNoCap);
		return;
	}
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	l.r_first = !g_track_on;
	if (!g_track_on) g_track.ResetAll(), g_track_on = true;
	l.shadow = g_track;
	l.shadow.ResetDraws();
	l.r_bad = p.cap_bad;
	Add(RWords, p.cap.size());
	const bool saved_bad = TejRepro::t_bad;
	TejRepro::t_acc = &l.shadow, TejRepro::t_cap = nullptr, TejRepro::Sync();
	const auto t = std::chrono::steady_clock::now();
	l.r_ok = LocalVulkanRecording::TejReplay(p.cap.data(), p.cap.size(), nullptr, LocalVulkanRecording::TejStubDispatch());
	l.shadow_ns = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count());
	if (TejRepro::t_bad) l.r_bad = true;
	TejRepro::t_bad = saved_bad;
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
	ReproFinish(l); // (tej-repro)
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
	ReproOpen(p); // (tej-repro: antes de Start, la sombra no entra en la sonda)
	Start();
}
inline void RenderDcbEnd() {
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
	const double f  = frames;
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
	            " pk/fot = %.0f draws/fot | REPRODUCIBLES (embudo + estado igual en cada draw) %.1f pk/fot = %.0f draws/fot; sin"
	            " handles %.1f pk/fot = %.0f draws/fot | iguales %.1f %% (sin handles %.1f %%) | distintos por motivo (%% de paquetes):"
	            " cuenta de draws %.1f, args de draw %.1f (sin handles %.1f), pipeline %.1f (%.1f), sets %.1f (%.1f), push %.1f (%.1f),"
	            " VB/IB %.1f (%.1f), estado dinamico %.1f (%.1f) | coste: reproducir %.3f us/draw, traducir ~%.3f us/draw (ventana"
	            " %.3f) | lista %.0f palabras/pk | AHORRO ESTIMADO paso B %.2f ms/fot (sin handles %.2f)\n",
	            rpk / f, v[RNoCap] / f, v[RFirst] / f, v[RBad] / f, v[RFail] / f, v[ROverflow] / f, v[RFunnel] / f,
	            v[RFunnelDraws] / f, v[ROk] / f, v[ROkDraws] / f, v[ROkNh] / f, v[ROkNhDraws] / f, rpc(v[REq]), rpc(v[REqNh]),
	            rpc(v[RDrawsDiff]), rpc(v[RArgs]), rpc(v[RArgsNh]), rpc(v[RCat0]), rpc(v[RNh0]), rpc(v[RCat1]), rpc(v[RNh1]),
	            rpc(v[RCat2]), rpc(v[RNh2]), rpc(v[RCat3]), rpc(v[RNh3]), rpc(v[RCat4]), rpc(v[RNh4]), rep_us, tra_us, win_us,
	            rpk > 0 ? static_cast<double>(v[RWords]) / rpk : 0.0, gain, gainnh);
}

} // namespace TejCoser
