#pragma once
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

namespace TejCoser {

[[nodiscard]] inline bool On() noexcept { return Wanted() && g_m7.load(std::memory_order_relaxed); }

// Lo que el trabajador deja en cada paquete (lo escribe antes de publicar: Slot::count con release).
struct PkInfo {
	Libs::Graphics::RenderState rs;
	bool                        rs_on   = false; // el buffer del trabajador estaba dentro de un rendering al abrir
	bool                        probed  = false;
	uint64_t                    end_key = ~uint64_t {0}; // primer draw del paquete siguiente (~0: fin de la DCB)
	Probe                       probe;
};

inline void WorkerOpen(PkInfo& p) {
	p.rs_on = false, p.probed = false, p.end_key = ~uint64_t {0};
	if (!On()) return;
	auto* rec = Libs::Graphics::CommandScheduler::ThreadRecorder();
	if (rec != nullptr && rec->command && !rec->command->IsInvalid() && rec->command->IsRendering()) {
		p.rs    = rec->command->CurrentRenderState();
		p.rs_on = true;
	}
	Start();
}
inline void WorkerClose(PkInfo& p) {
	if (!t_on) return;
	p.probe  = Stop();
	p.probed = true;
}

enum K : uint32_t {
	KPk, KDraws, KF1d, KNoHoles, KWValid, KWInRendering, KCompat, KHashEq, KHashEqSameDraws, KRValid, KAll, KAllDraws,
	KAbandon, KWBarrier, KWCopy, KWDispatch, KWBegin, KWEnd, KDiffDraws, KDiffArgs, KNoProbe, KDcbEnd, KN
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
};
inline Live g_live;

inline void Finish(bool dcb_end) {
	auto& l = g_live;
	if (!l.on) return;
	l.on            = false;
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
	else Add(rp.draws != l.w.draws ? KDiffDraws : KDiffArgs);
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
	Start();
}
inline void RenderDcbEnd() {
	if (!g_live.on) return;
	if (g_live.end_key == ~uint64_t {0}) Finish(true);
	else (void)Stop(), g_live.on = false, Add(KAbandon);
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
}

} // namespace TejCoser
