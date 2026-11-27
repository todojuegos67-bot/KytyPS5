#pragma once
// (techo) La linea [techo] cada 5 s y la tabla acumulada (local/techo.h). Hilo de render, en LiveControl::Flip.
// Ghidra (projelf_par1): CascadedShadows 0xa11220 -> RenderShadowCascade 0xa13e60 llama a 0xab57d0 (listas, marcas
// CENS), a 0xa57af0 (VirtualGeo) y a 0xab6080; SpotShadows 0xa10e20 ("RenderSingleSpotShadow"). 0xa57af0 y 0xab6080
// tambien salen de GBuffer 0xa05430, DepthPrePass 0xa0a370, Debug 0xa07e80, 0xa12860, 0xa12de0: por eso "fuera de
// marcas" se da por grupo (sombras / resto), no se inventa un reparto por via.
#include "local/techo.h"
#include "local/techo-faltas.h"
#include "local/techo-escena.h"
#include "local/normal-why.h"
#include "local/ds-censo-dcb.h"
#include "local/ds-censo-escena.h"
#include "local/live-census.h"
#include "local/live-counters.h"
#include "local-platform.h"

#include <array>

#include "local/bda-repite.h" // (bda-repite)

namespace Techo {

inline void Report(int64_t flip_ns, int64_t idle_ns_total) {
	if (!On()) return;
	using C = LiveCounters::Id;
	namespace LC = LiveCensus;
	struct Snap {
		int64_t  t = 0, idle = 0, upload_wait = 0;
		uint64_t tsc = 0, thread_cycles = 0;
		uint64_t kinds[LC::Kinds] {}, calls[LC::Kinds] {};
		uint64_t tab[2][LC::Phases] {}, disp[LC::Phases] {}, drawp[LC::Phases] {};
		int64_t  waits[LC::Waits] {};
		uint64_t site_ns[DsCensoDcb::MaxSites] {}, site_draws[DsCensoDcb::MaxSites] {}, site_disp[DsCensoDcb::MaxSites] {},
		         site_shadow[DsCensoDcb::MaxSites] {};
		uint64_t cens_ns[DsCensoEscena::Slots] {}, cens_draws[DsCensoEscena::Slots] {};
		uint64_t coh_cyc[Cohs][2] {}, coh_n[Cohs][2] {};
		uint64_t cnt[16] {};
	};
	static constexpr std::array counted {C::ProtectCalls, C::ProtectCallsRender, C::WriteFault, C::WindowFault, C::ReadFault,
	                                     C::Reprotect,    C::ReprotectPages,     C::UnprotectPages, C::RegionSyncUs,
	                                     C::BufferSyncUs, C::RegionSyncs,        C::SyncDownloads, C::TableDraws,
	                                     C::TableDispatches, C::UploadBytes,     C::TranslateUs};
	static_assert(counted.size() <= 16);
	static Snap     last;
	static uint64_t frames = 0;
	static int64_t  last_table = 0;
	static uint32_t window = 0;
	static bool     registered = false;
	if (!registered) {
		registered = true;
		std::printf("[techo] KYTY_TECHO=1: ms/fot del hilo de render por fase, una linea cada 5 s y tabla cada 30 s y al salir "
		            "(enciende tambien KYTY_CENSO_DCB, el censo de pases 0xab57d0 y los scopes de KYTY_RENDER_STATS; solo mide)\n");
		std::fflush(stdout);
		std::atexit([] { PrintTable("final"); });
	}
	++frames;
	BdaRepite::Flip(flip_ns); // (bda-repite)
	if (last.t != 0 && flip_ns - last.t < 5'000'000'000ll) return;
	Snap now;
	now.t             = flip_ns;
	now.idle          = idle_ns_total;
	now.upload_wait   = LC::g_upload_wait_ns.load(std::memory_order_relaxed);
	now.tsc           = __rdtsc();
	now.thread_cycles = LocalPlatform::CurrentThreadCycles();
	for (size_t i = 0; i < LC::Kinds; ++i) {
		now.kinds[i] = LC::g_kind_cycles[i];
		now.calls[i] = LC::g_kind_calls[i];
	}
	for (size_t i = 0; i < LC::Phases; ++i) {
		now.tab[0][i] = LC::g_table_phase_cycles[0][i];
		now.tab[1][i] = LC::g_table_phase_cycles[1][i];
		now.disp[i]   = LC::g_dispatch_phase_cycles[i];
		now.drawp[i]  = LC::g_draw_phase_cycles[i];
	}
	for (size_t i = 0; i < LC::Waits; ++i) now.waits[i] = LC::g_waits_ns[i];
	for (uint32_t i = 0; i < DsCensoDcb::MaxSites; ++i) {
		const auto& s     = DsCensoDcb::g_sites[i];
		now.site_ns[i]     = s.ns.load(std::memory_order_relaxed);
		now.site_draws[i]  = s.draws.load(std::memory_order_relaxed);
		now.site_disp[i]   = s.dispatches.load(std::memory_order_relaxed);
		now.site_shadow[i] = s.shadow_draws.load(std::memory_order_relaxed);
	}
	for (uint32_t i = 0; i < DsCensoEscena::Slots; ++i) {
		now.cens_ns[i]    = DsCensoEscena::g_acc[i].gpu_ns.load(std::memory_order_relaxed);
		now.cens_draws[i] = DsCensoEscena::g_acc[i].dcb_draws.load(std::memory_order_relaxed);
	}
	for (uint32_t k = 0; k < Cohs; ++k)
		for (uint32_t r = 0; r < 2; ++r) {
			now.coh_cyc[k][r] = g_coh_cyc[k][r].load(std::memory_order_relaxed);
			now.coh_n[k][r]   = g_coh_n[k][r].load(std::memory_order_relaxed);
		}
	for (size_t i = 0; i < counted.size(); ++i) now.cnt[i] = LiveCounters::Value(counted[i]);

	if (last.t != 0 && now.tsc > last.tsc && frames != 0) {
		++window;
		const double n       = static_cast<double>(frames);
		const double wall_ms = static_cast<double>(now.t - last.t) / 1e6;
		const double ms_cyc  = wall_ms / static_cast<double>(now.tsc - last.tsc); // ms por ciclo TSC
		const auto   cyc     = [&](uint64_t a, uint64_t b) { return static_cast<double>(a - b) * ms_cyc / n; };
		const auto   nsf     = [&](uint64_t a, uint64_t b) { return static_cast<double>(a - b) / 1e6 / n; };
		const auto   per     = [&](uint64_t a, uint64_t b) { return static_cast<double>(a - b) / n; };
		const auto   cnt     = [&](size_t i) { return per(now.cnt[i], last.cnt[i]); };
		const double frame   = wall_ms / n;
		const double idle    = static_cast<double>(now.idle - last.idle) / 1e6 / n;
		const double busy    = frame - idle;
		const bool   keep    = window >= 2; // (la 1a ventana medida es arranque/compilacion: fuera de la tabla)
		const auto   add     = [&](uint32_t block, const char* name, double ms, double extra) {
            if (keep) Accum(block, name, ms, extra, n);
		};
		// --- Pases: por sitio del juego (DCB/ACB) ---
		double shadow_sites = 0, other_sites = 0, top_ms[3] {};
		const char* top_name[3] {"", "", ""};
		static char names[DsCensoDcb::MaxSites][96];
		for (uint32_t i = 0; i < DsCensoDcb::MaxSites; ++i) {
			const double ms = nsf(now.site_ns[i], last.site_ns[i]);
			if (ms <= 0) continue;
			const uint64_t key = DsCensoDcb::g_sites[i].key.load(std::memory_order_relaxed);
			const uint32_t ret = static_cast<uint32_t>(key), tag = static_cast<uint32_t>(key >> 32u);
			const char*    nm  = i == DsCensoDcb::SiteAcb ? "ACB sin sitio" : i == DsCensoDcb::SiteUnknown ? "sin sitio"
			                                                                                             : DsCensoDcb::NameOf(ret);
			const char*    kind = (tag & 0xF0000000u) == DsCensoDcb::AcbTag ? "ACB" : "DCB";
			std::snprintf(names[i], sizeof(names[i]), "%06x %s#%u %.60s", ret, kind, tag & 0xffffu, nm != nullptr ? nm : "?");
			const double dd = per(now.site_draws[i], last.site_draws[i]) + per(now.site_disp[i], last.site_disp[i]);
			add(0, names[i], ms, dd);
			if (now.site_shadow[i] != last.site_shadow[i]) shadow_sites += ms; else other_sites += ms;
			for (int k = 0; k < 3; ++k)
				if (ms > top_ms[k]) {
					for (int j = 2; j > k; --j) { top_ms[j] = top_ms[j - 1]; top_name[j] = top_name[j - 1]; }
					top_ms[k] = ms; top_name[k] = names[i];
					break;
				}
		}
		// --- Dentro de pases: marcas CENS de 0xab57d0 ---
		double cens_shadow = 0, cens_scene = 0, casc = 0, spot = 0;
		for (uint32_t i = 0; i < DsCensoEscena::Slots; ++i) {
			const double ms = nsf(now.cens_ns[i], last.cens_ns[i]);
			if (ms <= 0) continue;
			char nm[96];
			std::snprintf(nm, sizeof(nm), "listas 0xab57d0 %s", DsCensoEscena::SlotName(i));
			add(1, nm, ms, per(now.cens_draws[i], last.cens_draws[i]));
			if (i >= DsCensoEscena::ShadowFirst) {
				cens_shadow += ms;
				(i < DsCensoEscena::ShadowFirst + 4 ? casc : spot) += ms;
			} else cens_scene += ms;
		}
		const double shadow_other = std::max(0.0, shadow_sites - cens_shadow), scene_other = std::max(0.0, other_sites - cens_scene);
		add(1, "sombras: otras vias (VirtualGeo 0xa57af0 + 0xab6080 + PM4 suelto de la DCB)", shadow_other, 0);
		add(1, "resto de DCB: fuera de listas 0xab57d0 (VirtualGeo, 0xab6080, dispatches, PM4)", scene_other, 0);
		// --- Coste por draw / dispatch ---
		const double xpr = cyc(now.kinds[LC::NativeXpr], last.kinds[LC::NativeXpr]);
		const double dsp = cyc(now.kinds[LC::Dispatch], last.kinds[LC::Dispatch]);
		const double drw = cyc(now.kinds[LC::Draw], last.kinds[LC::Draw]);
		const double sub = cyc(now.kinds[LC::Submission], last.kinds[LC::Submission]);
		const auto   td  = [&](size_t i) { return cyc(now.tab[0][i], last.tab[0][i]); };
		const auto   tq  = [&](size_t i) { return cyc(now.tab[1][i], last.tab[1][i]); };
		const double d_decode = td(1) + td(14), d_lookup = td(15) + cyc(now.kinds[LC::GraphicsPrograms], last.kinds[LC::GraphicsPrograms]);
		const double d_desc = td(2) + td(5) + td(6) + td(7), d_mem = td(3), d_rec = td(4) + td(9);
		const double q_decode = tq(1), q_desc = tq(2), q_mem = tq(3), q_rec = tq(4);
		const double draws_f = per(now.calls[LC::NativeXpr], last.calls[LC::NativeXpr]) + per(now.calls[LC::Draw], last.calls[LC::Draw]);
		const double disp_f  = per(now.calls[LC::Dispatch], last.calls[LC::Dispatch]);
		add(2, "draw: decodificar/estado (tabla 1 evaluar+gather, 14 clave de estado)", d_decode, draws_f);
		add(2, "draw: busqueda de shader/programa (tabla 15, GraphicsPrograms)", d_lookup, 0);
		add(2, "draw: descriptores/sets (tabla 2, 5-7 pruebas de set)", d_desc, 0);
		add(2, "draw: buffers+bloques, sync/subida de memoria (tabla 3, incl. BDA 12)", d_mem, 0);
		add(2, "draw: grabar Vulkan (tabla 4+9 emision)", d_rec, 0);
		add(2, "draw: camino normal (no tabla) entero", drw, per(now.calls[LC::Draw], last.calls[LC::Draw]));
		add(2, "dispatch: decodificar (tabla 1)", q_decode, disp_f);
		add(2, "dispatch: descriptores (tabla 2)", q_desc, 0);
		add(2, "dispatch: bloques/BDA sync (tabla 3)", q_mem, 0);
		add(2, "dispatch: grabar Vulkan (tabla 4)", q_rec, 0);
		add(2, "dispatch: resto fuera de tabla", std::max(0.0, dsp - q_decode - q_desc - q_mem - q_rec), 0);
		add(2, "PM4/registros fuera de draws y dispatches", std::max(0.0, sub - xpr - dsp - drw), 0);
		// --- Coherencia ---
		const double prot_r = cyc(now.coh_cyc[CProtect][1], last.coh_cyc[CProtect][1]), prot_g = cyc(now.coh_cyc[CProtect][0], last.coh_cyc[CProtect][0]);
		const double flt_r = cyc(now.coh_cyc[CFault][1], last.coh_cyc[CFault][1]), flt_g = cyc(now.coh_cyc[CFault][0], last.coh_cyc[CFault][0]);
		add(3, "proteccion (ProtectTransient) en hilo de render", prot_r, per(now.coh_n[CProtect][1], last.coh_n[CProtect][1]));
		add(3, "proteccion (ProtectTransient) en hilos del juego", prot_g, per(now.coh_n[CProtect][0], last.coh_n[CProtect][0]));
		add(3, "faltas (HandleFault) en hilo de render", flt_r, per(now.coh_n[CFault][1], last.coh_n[CFault][1]));
		add(3, "faltas (HandleFault) en hilos del juego", flt_g, per(now.coh_n[CFault][0], last.coh_n[CFault][0]));
		add(3, "sync de regiones BDA (RegionSyncUs, render)", cnt(8) / 1e3, cnt(10));
		add(3, "sync de buffers (BufferSyncUs, render)", cnt(9) / 1e3, 0);
		// --- Esperas ---
		static constexpr const char* wn[LC::Waits] = {"espera GPU", "espera readback", "descarga sincrona", "compilar shader/pipeline",
		                                               "espera worker de grabacion"};
		double waits = 0;
		for (size_t i = 0; i < LC::Waits; ++i) {
			const double ms = static_cast<double>(now.waits[i] - last.waits[i]) / 1e6 / n;
			waits += ms;
			add(4, wn[i], ms, 0);
		}
		add(4, "esperas de subida (cualquier hilo)", static_cast<double>(now.upload_wait - last.upload_wait) / 1e6 / n, 0);
		if (keep) {
			g_frames_total += n;
			g_busy_total += busy * n;
			g_frame_total += frame * n;
		}
		std::printf("[techo] v%u %.0f fot %.1f fps: fot %.2f ms, render ocupado %.2f (cpu %.2f), libre %.2f | top: %s %.2f, %s %.2f, %s %.2f"
		            " | sombras %.2f (cascadas 0xab57d0 %.2f, spot i+4 %.2f, otras vias %.2f) | escena fuera de listas %.2f"
		            " | por draw (%.0f draws, %.0f disp): decod %.2f shader %.2f desc %.2f mem %.2f vk %.2f | disp decod %.2f desc %.2f"
		            " mem %.2f vk %.2f | pm4 %.2f | coh render prot %.2f (%.0f) faltas %.2f (%.0f), juego prot %.2f (%.0f) faltas %.2f (%.0f),"
		            " protect_calls %.0f wfault %.0f wwin %.0f rfault %.0f reprot %.0f (pags %.0f) region_sync %.2f (%.0f) buf_sync %.2f"
		            " | esperas %.2f (gpu %.2f readback %.2f descarga %.2f compilar %.2f grabar %.2f)%s\n",
		            window, n, n * 1e3 / wall_ms, frame, busy,
		            now.thread_cycles != 0 && last.thread_cycles != 0 ? static_cast<double>(now.thread_cycles - last.thread_cycles) * ms_cyc / n : 0.0,
		            idle, top_name[0], top_ms[0], top_name[1], top_ms[1], top_name[2], top_ms[2], shadow_sites, casc, spot, shadow_other,
		            scene_other, draws_f, disp_f, d_decode, d_lookup, d_desc, d_mem, d_rec, q_decode, q_desc, q_mem, q_rec,
		            std::max(0.0, sub - xpr - dsp - drw), prot_r, per(now.coh_n[CProtect][1], last.coh_n[CProtect][1]), flt_r,
		            per(now.coh_n[CFault][1], last.coh_n[CFault][1]), prot_g, per(now.coh_n[CProtect][0], last.coh_n[CProtect][0]), flt_g,
		            per(now.coh_n[CFault][0], last.coh_n[CFault][0]), cnt(0), cnt(2), cnt(3), cnt(4), cnt(5), cnt(6), cnt(8) / 1e3, cnt(10),
		            cnt(9) / 1e3, waits, static_cast<double>(now.waits[0] - last.waits[0]) / 1e6 / n,
		            static_cast<double>(now.waits[1] - last.waits[1]) / 1e6 / n, static_cast<double>(now.waits[2] - last.waits[2]) / 1e6 / n,
		            static_cast<double>(now.waits[3] - last.waits[3]) / 1e6 / n, static_cast<double>(now.waits[4] - last.waits[4]) / 1e6 / n,
		            keep ? "" : " (arranque: fuera de la tabla)");
		std::fflush(stdout);
		TechoEscena::Window(busy, n * 1e3 / wall_ms, n, keep);    // (techo-escena, solo prueba)
		NormalWhy::Report(n);                                // (vg1, porque los draws van al camino normal)
		TechoFaltas::Report(n, ms_cyc); // (techo-faltas: the game threads' faults, a line beside this one)
		if (keep && flip_ns - last_table >= 30'000'000'000ll) {
			last_table = flip_ns;
			PrintTable("acumulada");
		}
	}
	last   = now;
	frames = 0;
}

} // namespace Techo
