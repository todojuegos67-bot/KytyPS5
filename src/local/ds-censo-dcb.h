#pragma once
// (w7) KYTY_CENSO_DCB=1 (apagado por defecto; Demon's Souls 01.007.000): CENSO por DCB del hilo de render (hilo de la
// GPU del emulador). Solo mide. Responde: en que bloques del juego se van los ms por frame que no son mallas.
//
// Ghidra (projelf_par1): toda DCB grafica del juego sale por la envoltura de libAgc 0x1d09fb0 (push rbp; mov rbp,rsp;
// push rbx; sub rsp,0x18; ... lea rdi,[rbp-0x20]; call sceAgcDriverSubmitDcb) => en AgcDriverSubmitDcb(packet):
// rbp del juego = packet + 0x20, y la pila sigue la cadena de rbp (todas las funciones de la ruta tienen push rbp/mov rbp,rsp).
// Ruta: 0x1d09fb0 <- CRenderContext::Submit 0x911c90 / SubmitAndFlip 0x911d60 <- 0xa18840 <- RenderThread::KickoffSubmitJob
// 0xa08ae0 <- EL SITIO (la funcion de render que pidio el envio). Tambien BinkGPU 0x9808a0 (video) llama a 0x1d09fb0.
// Sitio = primera direccion de retorno de la cadena que no cae en esas envolturas; padre = la siguiente.
//
// Hilo del juego (submit_dcb de agc.cpp): sitio + padre, draws y dispatches de la DCB (cabeceras PM4 de tipo 3), en un
// anillo indexado por la direccion de la DCB. Hilo de la GPU (GuestGpu::Process): ns de cada trozo de cada submission
// (cola 0 por sitio; colas de computo aparte), ns del barrido SweepCpuWrites aparte, y ns de pared entre fronteras de frame.
// Linea "[censo-dcb]" cada 5 s, ordenada por us/frame: sitio, nombre, DCB/f, draws/f, dispatches/f, us/f.
// "fuera de Process" = pared del hilo - Process: esperas del hilo de la GPU (cola vacia, flip) y lo que no es traduccion.
//
// (w8) Ghidra: con el envio asincrono (*(params+0x68)+0xc9f != 0) KickoffSubmitJob 0xa08ae0 no envia: copia la lista de
// contextos y lanza el trabajo "SubmitJob" (0xa18840) en el sistema de jobs. El trabajador 0x83c1e0 (bucle 0x83b520) lo
// ejecuta con "call rax" en 0x83c461 => la pila de rbp acaba en 0x83c463 y se pierde quien pidio el envio. Por eso:
//  - Gancho en la ENTRADA de 0xa08ae0 (loader/demonsSoulsNativoSombras.inc, solo con KYTY_CENSO_DCB=1): G =
//    *(*(eboot+0x2f65f58)+0x10). DCB: contextos *(G+0x10)[G.i32[0] .. G.i32[1] - (modo != 1)), dcb = **contexto
//    (CRenderContext::Submit 0x911c90: m_dcb.getSubmitPointer()). ACB: *(G+0xf8)[G.i32[2] .. G.i32[3]), acb = *elem
//    (0x915540 -> 0x1d0a090 -> sceAgcDriverSubmitAcb: la UNICA ruta de ACB del juego). Se apunta direccion -> (sitio que
//    llamo a KickoffSubmitJob, n de orden dentro del envio) antes de ejecutar la funcion original.
//  - Dentro de cada DCB: las marcas CENS del gancho 0xab57d0 (ds-censo-escena.h; con w8 tambien las sombras, pase >= 0)
//    reparten los draws en sombras / mallas del escenario / resto; y los NOP con texto ASCII (marcadores del juego) se
//    cuentan y se guarda el primero.
//  - ACB por sitio igual (AgcDriverSubmitAcb / SubmitMultiAcbs -> OnSubmitAcb), con su tiempo en el hilo de la GPU.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "local/techo.h" // (techo: KYTY_TECHO lo enciende)

namespace DsCensoDcb {

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* t = std::getenv("KYTY_CENSO_DCB");
		return Techo::On() || (t != nullptr && t[0] == '1' && t[1] == '\0');
	}();
	return on;
}

// --- Nombres (Ghidra, projelf_par1; offsets del eboot 01.007.000 = direccion de retorno) ---
struct Name { uint32_t ret; const char* name; };
inline constexpr Name kNames[] = {
    {0xa0decd, "a0de50 pre-frame (PreCulling / GPULightCullingJob / XPRPreRender)"},
    {0xa0ed3e, "a0eb80 forward #1 (tras VirtualGeometry::PreRenderJob)"},
    {0xa0ed8e, "a0eb80 forward #2"},
    {0xa0ee77, "a0eb80 forward #3 (antes de CreateClusters/Fur/Lightmaps/Fog/Standard/Transparent)"},
    {0xa0913d, "a090e0 diferido #1 (antes de RenderDepth)"},
    {0xa0937a, "a090e0 diferido #2 (RenderDepth -> GBuffer/Decal)"},
    {0xa09a7b, "a090e0 diferido #3 (tras RenderDecal: PostDepthAsync/Fur/Skybox)"},
    {0xa08ad5, "a08ab0"},
    {0xa12393, "a12350 #1"}, {0xa12597, "a12350 #2"}, {0xa12707, "a12350 #3"},
    {0xa150fa, "a14886"}, {0xa16bf9, "a16a20"},
    {0x97f1b7, "BinkGPU (video) 97ee90 #1"}, {0x97f20d, "BinkGPU (video) 97ee90 #2"},
};
[[nodiscard]] inline const char* NameOf(uint32_t ret) {
	for (const auto& n: kNames)
		if (n.ret == ret) return n.name;
	return nullptr;
}
// Envolturas que se saltan al buscar el sitio (rangos de funcion de Ghidra).
[[nodiscard]] inline bool IsWrapper(uint64_t off) {
	return (off >= 0x1d09fb0 && off < 0x1d0a024) || (off >= 0x911c90 && off < 0x911e55) || (off >= 0xa18840 && off < 0xa1898b) ||
	       (off >= 0xa08ae0 && off < 0xa090d1) || (off >= 0x9808a0 && off < 0x981088);
}

// --- Sitios ---
constexpr uint32_t MaxSites = 128;
constexpr uint32_t CensoShadowFirst = 16; // (= DsCensoEscena::ShadowFirst: pase de las marcas CENS de sombras)
constexpr uint32_t SiteAcb = MaxSites - 1, SiteUnknown = MaxSites - 2; // (colas de computo sin sitio; sin pila reconocida)
// (w8) clave = ret | etiqueta << 32. Etiqueta: padre (pila) o KickTag | orden (gancho de 0xa08ae0, DCB) o AcbTag | orden.
constexpr uint64_t KickTag = 0xD0000000u, AcbTag = 0xA0000000u;
struct Site {
	std::atomic<uint64_t> key {0}; // ret | padre << 32 (0 = libre)
	std::atomic<uint64_t> dcbs {0}, draws {0}, dispatches {0}, dwords {0}; // hilo del juego
	std::atomic<uint64_t> ns {0}, slices {0};                               // hilo de la GPU
	std::atomic<uint64_t> shadow_draws {0}, scene_draws {0}, markers {0};   // (w8) dentro de la DCB
	std::atomic<uint64_t> marker_text[4] {};                                // (w8) primer marcador ASCII (32 bytes)
};
inline Site g_sites[MaxSites];
inline std::atomic<uint64_t> g_base {0};
inline std::atomic<uint64_t> g_no_stack {0};

[[nodiscard]] inline uint32_t SiteFor(uint64_t key) {
	for (uint32_t i = 0; i < SiteUnknown; ++i) {
		uint64_t k = g_sites[i].key.load(std::memory_order_acquire);
		if (k == key) return i;
		if (k == 0) {
			if (g_sites[i].key.compare_exchange_strong(k, key, std::memory_order_acq_rel)) return i;
			if (k == key) return i;
		}
	}
	return SiteUnknown;
}

// --- (w8) Mapa direccion -> clave, escrito en la entrada de KickoffSubmitJob (hilo del juego) ---
struct KickSlot { std::atomic<uint64_t> addr {0}, key {0}; };
constexpr uint32_t KickSize = 1024;
inline KickSlot g_kick[KickSize];
inline std::atomic<uint32_t> g_kick_at {0};
inline std::atomic<uint64_t> g_kicks {0}, g_kick_dcbs {0}, g_kick_acbs {0};
inline void KickNote(uint64_t addr, uint64_t key) {
	if (addr == 0) return;
	const uint32_t at = g_kick_at.fetch_add(1, std::memory_order_acq_rel);
	g_kick[at % KickSize].addr.store(0, std::memory_order_relaxed);
	g_kick[at % KickSize].key.store(key, std::memory_order_relaxed);
	g_kick[at % KickSize].addr.store(addr, std::memory_order_release);
}
[[nodiscard]] inline uint64_t KickLookup(uint64_t addr) {
	const uint32_t at = g_kick_at.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < KickSize && i < at; ++i) {
		const KickSlot& k = g_kick[(at - 1u - i) % KickSize];
		if (k.addr.load(std::memory_order_acquire) == addr) return k.key.load(std::memory_order_relaxed);
	}
	return 0;
}
// Gancho de 0xa08ae0 (renderer, params, modo): ret = sitio que pidio el envio (desplazamiento en el eboot).
inline void OnKickoff(uint64_t base, uint64_t ret_off, int32_t mode) {
	const uint64_t holder = *reinterpret_cast<const uint64_t*>(base + 0x2f65f58);
	if (holder == 0) return;
	const uint64_t g = *reinterpret_cast<const uint64_t*>(holder + 0x10);
	if (g == 0) return;
	const auto* gi = reinterpret_cast<const int32_t*>(g);
	g_kicks.fetch_add(1, std::memory_order_relaxed);
	const uint64_t site = ret_off & 0xffffffffu;
	const auto* dl = *reinterpret_cast<const uint64_t* const*>(g + 0x10);
	const int32_t d0 = gi[0], d1 = gi[1] - (mode != 1 ? 1 : 0);
	for (int32_t i = d0; dl != nullptr && i < d1 && i - d0 < 64; ++i) {
		const uint64_t elem = dl[i];
		if (elem == 0) continue;
		const uint64_t x = *reinterpret_cast<const uint64_t*>(elem);
		if (x == 0) continue;
		KickNote(*reinterpret_cast<const uint64_t*>(x), site | ((KickTag | uint64_t(i - d0)) << 32u));
		g_kick_dcbs.fetch_add(1, std::memory_order_relaxed);
	}
	const auto* al = *reinterpret_cast<const uint64_t* const*>(g + 0xf8);
	for (int32_t i = gi[2]; al != nullptr && i < gi[3] && i - gi[2] < 64; ++i) {
		const uint64_t elem = al[i];
		if (elem == 0) continue;
		KickNote(*reinterpret_cast<const uint64_t*>(elem), site | ((AcbTag | uint64_t(i - gi[2])) << 32u));
		g_kick_acbs.fetch_add(1, std::memory_order_relaxed);
	}
}

// --- Anillo DCB -> sitio ---
struct Slot { std::atomic<uint64_t> dcb {0}; std::atomic<uint32_t> site {SiteUnknown}; };
constexpr uint32_t RingSize = 512;
inline Slot g_ring[RingSize], g_ring_acb[RingSize];
inline std::atomic<uint32_t> g_ring_at {0}, g_ring_acb_at {0};

[[nodiscard]] inline uint32_t LookupIn(const Slot* ring, const std::atomic<uint32_t>& ring_at, const void* dcb, uint32_t fallback) {
	const auto p = reinterpret_cast<uint64_t>(dcb);
	const uint32_t at = ring_at.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < RingSize; ++i) { // (de la mas reciente hacia atras)
		const Slot& s = ring[(at - 1u - i) % RingSize];
		if (s.dcb.load(std::memory_order_acquire) == p) return s.site.load(std::memory_order_relaxed);
	}
	return fallback;
}
[[nodiscard]] inline uint32_t LookupDcb(const void* dcb) { return LookupIn(g_ring, g_ring_at, dcb, SiteUnknown); }
[[nodiscard]] inline uint32_t LookupAcb(const void* acb) { return LookupIn(g_ring_acb, g_ring_acb_at, acb, SiteAcb); }

// (w8) Recorre la DCB: draws / dispatches; draws dentro de marcas CENS (sombras: pase de sombras; escenario: el resto);
// NOP con texto ASCII (marcadores del juego).
inline void ScanCommands(Site& s, const uint32_t* dcb, uint32_t dw) {
	uint64_t draws = 0, dispatches = 0, shadow = 0, scene = 0, markers = 0;
	int open = -1; // (-1 fuera; 0 escenario; 1 sombras)
	for (uint32_t i = 0; i < dw;) {
		const uint32_t h = dcb[i];
		if ((h >> 30u) != 3u) { ++i; continue; }
		const uint32_t len = ((h >> 16u) & 0x3fffu) + 2u;
		switch ((h >> 8u) & 0xffu) {
			case 0x24: case 0x25: case 0x27: case 0x2C: case 0x2D: case 0x35: case 0x38: case 0x8D:
				++draws;
				if (open == 1) ++shadow; else if (open == 0) ++scene;
				break;
			case 0x15: case 0x16: ++dispatches; break;
			case 0x10:
				if (len == 6u && i + 6u <= dw && dcb[i + 1] == 0x424d4f53u && dcb[i + 2] == 0x534e4543u) { // ("SOMB" "CENS")
					const uint32_t kind = dcb[i + 3] & 0xffu, pass = (dcb[i + 3] >> 8u) & 0xffu;
					open = kind == 1u ? (pass >= CensoShadowFirst ? 1 : 0) : -1;
				} else if (len >= 3u && i + len <= dw) {
					const auto* c = reinterpret_cast<const unsigned char*>(dcb + i + 1);
					const uint32_t bytes = std::min<uint32_t>((len - 1u) * 4u, 32u);
					uint32_t printable = 0;
					while (printable < bytes && c[printable] >= 0x20 && c[printable] < 0x7f) ++printable;
					if (printable >= 3 && (printable == bytes || c[printable] == 0)) {
						++markers;
						if (s.marker_text[0].load(std::memory_order_relaxed) == 0) {
							uint64_t t[4] {};
							std::memcpy(t, c, std::min<uint32_t>(printable, 31u));
							for (int k = 3; k >= 0; --k) s.marker_text[k].store(t[k], std::memory_order_relaxed);
						}
					}
				}
				break;
			default: break;
		}
		i += len;
	}
	s.draws.fetch_add(draws, std::memory_order_relaxed);
	s.dispatches.fetch_add(dispatches, std::memory_order_relaxed);
	s.shadow_draws.fetch_add(shadow, std::memory_order_relaxed);
	s.scene_draws.fetch_add(scene, std::memory_order_relaxed);
	s.markers.fetch_add(markers, std::memory_order_relaxed);
}

// Hilo del juego, en AgcDriverSubmitDcb (hle_ret = __builtin_return_address(0); packet = su argumento).
inline void OnSubmit(uint64_t hle_ret, const void* packet, const uint32_t* dcb, uint32_t dw) {
	uint64_t base = g_base.load(std::memory_order_relaxed);
	if (base == 0) { // (retorno a 0x1d09fff o 0x1d0a00e; la base va alineada a 16 KiB)
		for (const uint64_t r: {uint64_t {0x1d09fff}, uint64_t {0x1d0a00e}}) {
			const uint64_t b = hle_ret - r;
			if ((b & 0x3fffu) == 0 && std::memcmp(reinterpret_cast<const void*>(b + 0x1d09fb0), "\x55\x48\x89\xe5\x53\x48\x83\xec\x18", 9) == 0) {
				base = b;
				g_base.store(b, std::memory_order_relaxed);
				std::printf("[censo-dcb] eboot en 0x%llx (envoltura 0x1d09fb0 reconocida)\n", (unsigned long long)b);
				std::fflush(stdout);
				break;
			}
		}
	}
	uint32_t site = SiteUnknown;
	const uint64_t ro = hle_ret - base;
	if (const uint64_t kk = KickLookup(reinterpret_cast<uint64_t>(dcb)); kk != 0) {
		site = SiteFor(kk); // (w8: el sitio que llamo a KickoffSubmitJob y el orden de la DCB en ese envio)
	} else if (base != 0 && (ro == 0x1d09fff || ro == 0x1d0a00e)) {
		uint64_t rbp = reinterpret_cast<uint64_t>(packet) + 0x20u, found = 0, parent = 0;
		for (int depth = 0; depth < 10 && rbp != 0; ++depth) {
			const uint64_t ret = reinterpret_cast<const uint64_t*>(rbp)[1] - base;
			if (!IsWrapper(ret)) {
				if (found == 0) found = ret; else { parent = ret; break; }
			}
			const uint64_t next = reinterpret_cast<const uint64_t*>(rbp)[0];
			if (next <= rbp || next - rbp > (1u << 20u) || (next & 7u) != 0) break; // (cadena rota: paramos)
			rbp = next;
		}
		if (found != 0 && found < (1ull << 32u)) site = SiteFor(found | ((parent < (1ull << 32u) ? parent : 0) << 32u));
	}
	if (site == SiteUnknown) g_no_stack.fetch_add(1, std::memory_order_relaxed);
	Site& s = g_sites[site];
	s.dcbs.fetch_add(1, std::memory_order_relaxed);
	ScanCommands(s, dcb, dw);
	s.dwords.fetch_add(dw, std::memory_order_relaxed);
	const uint32_t at = g_ring_at.load(std::memory_order_relaxed);
	g_ring[at % RingSize].site.store(site, std::memory_order_relaxed);
	g_ring[at % RingSize].dcb.store(reinterpret_cast<uint64_t>(dcb), std::memory_order_release);
	g_ring_at.store(at + 1u, std::memory_order_release); // (un solo hilo de render del juego envia DCB graficas)
}

// (w8) Hilo del juego, en AgcDriverSubmitAcb / SubmitMultiAcbs.
inline void OnSubmitAcb(const uint32_t* acb, uint32_t dw) {
	const uint64_t kk = KickLookup(reinterpret_cast<uint64_t>(acb));
	const uint32_t site = kk != 0 ? SiteFor(kk) : SiteAcb;
	Site& s = g_sites[site];
	s.dcbs.fetch_add(1, std::memory_order_relaxed);
	ScanCommands(s, acb, dw);
	s.dwords.fetch_add(dw, std::memory_order_relaxed);
	const uint32_t at = g_ring_acb_at.fetch_add(1, std::memory_order_acq_rel);
	g_ring_acb[at % RingSize].site.store(site, std::memory_order_relaxed);
	g_ring_acb[at % RingSize].dcb.store(reinterpret_cast<uint64_t>(acb), std::memory_order_release);
}

// --- Hilo de la GPU ---
inline std::atomic<uint64_t> g_sweep_ns {0}, g_process_ns {0}, g_wall_ns {0}, g_frames {0};
[[nodiscard]] inline uint64_t NowNs() {
	return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
// Un trozo de submission (Process): RAII alrededor del cuerpo de GuestGpu::Process.
struct Slice {
	uint32_t site;
	uint64_t t0 = NowNs();
	Slice(bool graphics_queue0, const void* dcb) : site(graphics_queue0 ? LookupDcb(dcb) : LookupAcb(dcb)) {}
	~Slice() {
		const uint64_t d = NowNs() - t0;
		g_sites[site].ns.fetch_add(d, std::memory_order_relaxed);
		g_sites[site].slices.fetch_add(1, std::memory_order_relaxed);
		g_process_ns.fetch_add(d, std::memory_order_relaxed);
	}
	Slice(const Slice&) = delete;
	Slice& operator=(const Slice&) = delete;
};

inline void Report() {
	static uint64_t last_print = NowNs();
	static uint64_t prev[MaxSites][9] {};
	static uint64_t prev_sweep = 0, prev_proc = 0, prev_wall = 0, prev_frames = 0, prev_nostack = 0;
	const uint64_t now = NowNs();
	if (now - last_print < 5000000000ull) return;
	last_print = now;
	const uint64_t frames_total = g_frames.load(), frames = std::max<uint64_t>(1, frames_total - prev_frames);
	prev_frames = frames_total;
	const double f = double(frames);
	const uint64_t sweep = g_sweep_ns.load() - prev_sweep, proc = g_process_ns.load() - prev_proc, wall = g_wall_ns.load() - prev_wall;
	prev_sweep += sweep; prev_proc += proc; prev_wall += wall;
	const uint64_t nostack = g_no_stack.load() - prev_nostack;
	prev_nostack += nostack;
	struct Row { uint32_t i; uint64_t v[9]; };
	Row rows[MaxSites];
	uint32_t n = 0;
	for (uint32_t i = 0; i < MaxSites; ++i) {
		const uint64_t cur[9] = {g_sites[i].ns.load(), g_sites[i].dcbs.load(), g_sites[i].draws.load(), g_sites[i].dispatches.load(),
		                         g_sites[i].dwords.load(), g_sites[i].slices.load(), g_sites[i].shadow_draws.load(),
		                         g_sites[i].scene_draws.load(), g_sites[i].markers.load()};
		Row r {i, {}};
		bool any = false;
		for (int k = 0; k < 9; ++k) { r.v[k] = cur[k] - prev[i][k]; prev[i][k] = cur[k]; any |= r.v[k] != 0; }
		if (any) rows[n++] = r;
	}
	std::sort(rows, rows + n, [](const Row& a, const Row& b) { return a.v[0] > b.v[0]; });
	std::printf("[censo-dcb] %llu frames | hilo GPU: pared %.2f ms/f, Process %.2f ms/f (de ello barrido SweepCpuWrites %.2f), "
	            "fuera de Process %.2f ms/f | DCB sin pila reconocida %.1f/f\n",
	            (unsigned long long)frames, double(wall) / 1e6 / f, double(proc) / 1e6 / f, double(sweep) / 1e6 / f,
	            (double(wall) - double(proc)) / 1e6 / f, double(nostack) / f);
	std::printf("[censo-dcb] (w8) gancho KickoffSubmitJob 0xa08ae0: %llu envios, %llu DCB y %llu ACB apuntadas (total)\n",
	            (unsigned long long)g_kicks.load(), (unsigned long long)g_kick_dcbs.load(), (unsigned long long)g_kick_acbs.load());
	for (uint32_t k = 0; k < n; ++k) {
		const Row& r = rows[k];
		const uint64_t key = g_sites[r.i].key.load();
		const uint32_t ret = uint32_t(key), parent = uint32_t(key >> 32u);
		const bool kick = (parent & 0xF0000000u) == KickTag, acb = (parent & 0xF0000000u) == AcbTag;
		const char* nm = r.i == SiteAcb ? "colas de computo (ACB) sin sitio" : r.i == SiteUnknown ? "sin sitio" : NameOf(ret);
		char tag[48];
		if (kick) std::snprintf(tag, sizeof(tag), "DCB #%u del envio", parent & 0xffffu);
		else if (acb) std::snprintf(tag, sizeof(tag), "ACB #%u del envio", parent & 0xffffu);
		else std::snprintf(tag, sizeof(tag), "padre 0x%06x", parent);
		uint64_t mt[4];
		for (int q = 0; q < 4; ++q) mt[q] = g_sites[r.i].marker_text[q].load(std::memory_order_relaxed);
		char mk[33] {};
		std::memcpy(mk, mt, 32);
		const double other = double(r.v[2]) - double(r.v[6]) - double(r.v[7]);
		std::printf("[censo-dcb]   %6.0f us/f | eboot+0x%06x (%s) %s | %.1f buf/f %.0f draws/f (sombras %.0f, mallas escenario %.0f, "
		            "resto %.0f) %.0f dispatches/f %.0f KiB/f %.1f trozos/f | marcadores %.1f/f primero \"%s\"\n",
		            double(r.v[0]) / 1e3 / f, ret, tag, nm != nullptr ? nm : "?", double(r.v[1]) / f, double(r.v[2]) / f,
		            double(r.v[6]) / f, double(r.v[7]) / f, other / f, double(r.v[3]) / f, double(r.v[4]) * 4.0 / 1024.0 / f,
		            double(r.v[5]) / f, double(r.v[8]) / f, mk);
	}
	std::fflush(stdout);
}

// Hilo de la GPU, en la frontera de frame (SubmissionType::FrameBoundary de GuestGpu::Process).
inline void OnFrameBoundary() {
	static uint64_t last = 0;
	const uint64_t now = NowNs();
	if (last != 0) g_wall_ns.fetch_add(now - last, std::memory_order_relaxed);
	last = now;
	g_frames.fetch_add(1, std::memory_order_relaxed);
	Report();
}

} // namespace DsCensoDcb
