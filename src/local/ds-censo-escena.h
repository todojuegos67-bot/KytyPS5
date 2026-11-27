#pragma once
// (F5 pase 3, pieza 0) KYTY_NATIVO_PASE_ESCENA=2 (apagado por defecto; Demon's Souls 01.007.000): CENSO por pase del
// dibujo del escenario y los personajes. Solo mide; =1 / =3 (dibujo nativo) aun no existen y se tratan como =2.
//
// Ghidra (projelf_par1): todos los pases de mallas pasan por 0xab57d0(renderer, lista, ctx, cubo, pase) (el mismo
// gancho que las sombras; las sombras son pase >= 0, el escenario pase < 0). Quien la llama decide el pase:
//   0xa090e0 renderer diferido: "RenderDepth" / "RenderGBuffer" / "RenderDecal" / "SkyboxOpaque" / "SkyboxAlpha"
//     0xa0a370 "DepthPrePass" (prepase de profundidad)      0xa05430 "GBuffer" (pase de material, el mas caro esperado)
//     0xa06520 (desde 0xa11e70 y 0xa090e0)                  0xa11e70 "DepthPrePass" (desde 0x9ab420)
//   0xa0eb80 forward: "Standard" "ForwardDecal" "Transparent" (+ jobs de VirtualGeometry, CreateClusters de luces,
//     Fur, UpdateLightmaps, Fog)
//     0xa06880 "Water"  0xa06ba0 "Transparent"  0xa07580 "DepthPrePass"/"AlphaForceNoMotion"/"Transparent"
//     0xa07e80 "Debug"  0xa08180 (forward)  0xa24c50 "GpuPicking"
//   0xa12de0 / 0xa13110 ("DepthPrePass") / 0xa13320: desde 0xa10ec2 / 0xa0e9c0 / 0xa0e7f0 (vistas extra: reflejo/cubo)
//   0xa0e4d0 (desde 0xa0de50)
// Las luces de antorchas y almenaras NO pasan por aqui: son el CreateClustersJob del forward (dispatches) y las
// particulas de "Transparent"; este censo las separa del resto (pase Transparent / Standard frente a GBuffer).
//
// Hilo del juego (gancho de 0xab57d0 con pase < 0): pone una marca CENS de 6 dwords delante y otra detras de los draws
// de la llamada (sitio de llamada = direccion de retorno - eboot), cuenta entradas, tecnicas distintas y draws de la DCB.
// Hilo de la GPU (graphicsRun.cpp): entre la marca de inicio y la de fin mide el tiempo de traduccion (steady_clock) y lo
// suma al pase. Linea "[censo-escena]" cada 5 s: por pase draws/frame, us/frame, us/draw y entradas/tecnicas.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "local/ds-censo-dcb.h" // (w8: DsCensoDcb::OnKickoff, desde el mismo .inc del gancho)

namespace DsCensoEscena {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* t = std::getenv("KYTY_NATIVO_PASE_ESCENA");
		if (Techo::On() && (t == nullptr || t[0] == '\0' || t[0] == '0')) return 2u; // (techo)
		if (t == nullptr || t[0] == '\0' || t[1] != '\0') return 0u;
		return (t[0] == '1' || t[0] == '2' || t[0] == '3') ? 2u : 0u; // (solo mide por ahora)
	}();
	return mode;
}

constexpr uint32_t MarkDwords = 6; // cabecera, "SOMB", "CENS", tipo | pase<<8, seq, draws de la DCB (en la de fin)
constexpr uint32_t MarkHeader = 0xC0000000u | ((MarkDwords - 2u) << 16u) | (0x10u << 8u); // (= DsNativoSombras::MarkHeader)
constexpr uint32_t Magic0 = 0x424d4f53u, Magic1 = 0x534e4543u; // "SOMB" "CENS"
constexpr uint32_t Begin = 1, End = 2;
[[nodiscard]] inline bool IsMark(const uint32_t* p, uint32_t remaining) {
	return remaining >= MarkDwords && p[0] == MarkHeader && p[1] == Magic0 && p[2] == Magic1;
}

// Los pases: la funcion que llama a 0xab57d0 (entrada, en el eboot) y su nombre (cadenas de Ghidra).
struct Pass { uint32_t entry, end; const char* name; };
inline constexpr Pass kPasses[] = {
    {0xa05430, 0xa05703, "GBuffer"},           {0xa06520, 0xa06707, "a06520(Depth/GBuf)"},
    {0xa06880, 0xa06a23, "Water"},             {0xa06ba0, 0xa07118, "Transparent(a06ba0)"},
    {0xa07580, 0xa079e1, "Forward(Depth/AlphaNoMotion/Transp)"}, {0xa07e80, 0xa0817b, "Debug"},
    {0xa08180, 0xa08425, "Forward(a08180)"},  {0xa0a370, 0xa0a627, "DepthPrePass"},
    {0xa0e4d0, 0xa0e68d, "a0e4d0"},            {0xa0eb80, 0xa10e13, "Forward Standard/Decal"},
    {0xa11e70, 0xa121ed, "DepthPrePass(a11e70)"}, {0xa12de0, 0xa13109, "Vista extra a12de0"},
    {0xa13110, 0xa1331b, "Vista extra DepthPrePass"}, {0xa13320, 0xa1354c, "Vista extra a13320"},
    {0xa24c50, 0xa25123, "GpuPicking"},
};
constexpr uint32_t PassCount = sizeof(kPasses) / sizeof(kPasses[0]);
constexpr uint32_t Other = PassCount; // (sitio de llamada fuera de la tabla: wrappers 0xa063c0.. etc.)
// (w8) Sombras (pase >= 0 de 0xab57d0: CascadedShadows 0xa11220 -> 0xa13e60, cascada i y sus extra i+4), con
// KYTY_NATIVO_PASE_SOMBRAS apagado: ShadowFirst + min(pase, 7).
constexpr uint32_t ShadowFirst = PassCount + 1, ShadowSlots = 8, Slots = ShadowFirst + ShadowSlots;
static_assert(ShadowFirst == DsCensoDcb::CensoShadowFirst);
inline constexpr const char* kShadowNames[ShadowSlots] = {"Sombras cascada 0", "Sombras cascada 1", "Sombras cascada 2",
    "Sombras cascada 3", "Sombras extra 4", "Sombras extra 5", "Sombras extra 6", "Sombras extra 7"};
[[nodiscard]] inline const char* SlotName(uint32_t i) {
	return i < PassCount ? kPasses[i].name : i == Other ? "otro" : i < Slots ? kShadowNames[i - ShadowFirst] : "?";
}
[[nodiscard]] inline uint32_t PassOf(uint64_t ret_offset) {
	for (uint32_t i = 0; i < PassCount; ++i)
		if (ret_offset >= kPasses[i].entry && ret_offset < kPasses[i].end) return i;
	return Other;
}

struct Acc {
	std::atomic<uint64_t> calls {0}, entries {0}, techniques {0}, dcb_draws {0}; // hilo del juego
	std::atomic<uint64_t> gpu_ns {0}, gpu_spans {0};                            // hilo de la GPU
};
inline Acc g_acc[Slots];
inline std::atomic<uint32_t> g_seq {0};
inline std::atomic<uint64_t> g_last_other {0}; // (ultimo sitio de llamada sin nombre, para ampliar la tabla)

// --- Hilo de la GPU ---
inline thread_local std::chrono::steady_clock::time_point g_begin_at {};
inline thread_local uint32_t g_begin_pass = 0xFFFFFFFFu, g_begin_seq = 0;
inline void GpuMark(const uint32_t* p) {
	const uint32_t kind = p[3] & 0xFFu, pass = (p[3] >> 8u) & 0xFFu;
	const auto now = std::chrono::steady_clock::now();
	if (kind == Begin) { g_begin_at = now; g_begin_pass = pass; g_begin_seq = p[4]; return; }
	if (kind == End && pass == g_begin_pass && p[4] == g_begin_seq && pass < Slots) {
		g_acc[pass].gpu_ns.fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_begin_at).count()),
		                             std::memory_order_relaxed);
		g_acc[pass].gpu_spans.fetch_add(1, std::memory_order_relaxed);
	}
	g_begin_pass = 0xFFFFFFFFu;
}

// Una linea cada 5 s (la llama el hilo de la GPU en cada flip o marca). frames = frames presentados desde la anterior.
inline void Report(uint64_t frames_total) {
	using Clock = std::chrono::steady_clock;
	static Clock::time_point last = Clock::now();
	static uint64_t last_frames = frames_total;
	static uint64_t prev[Slots][3] {};
	const auto now = Clock::now();
	if (now - last < std::chrono::seconds(5)) return;
	last = now;
	const uint64_t frames = std::max<uint64_t>(1, frames_total - last_frames);
	last_frames = frames_total;
	struct Row { uint32_t i; uint64_t draws, ns, entries; };
	Row rows[Slots];
	uint64_t total_draws = 0, total_ns = 0;
	for (uint32_t i = 0; i < Slots; ++i) {
		const uint64_t d = g_acc[i].dcb_draws.load(std::memory_order_relaxed), n = g_acc[i].gpu_ns.load(std::memory_order_relaxed),
		               e = g_acc[i].entries.load(std::memory_order_relaxed);
		rows[i] = {i, d - prev[i][0], n - prev[i][1], e - prev[i][2]};
		prev[i][0] = d; prev[i][1] = n; prev[i][2] = e;
		total_draws += rows[i].draws; total_ns += rows[i].ns;
	}
	std::sort(rows, rows + Slots, [](const Row& a, const Row& b) { return a.ns > b.ns; });
	char line[4096];
	size_t at = size_t(std::snprintf(line, sizeof(line), "[censo-escena] %llu frames | total %.0f draws/frame %.0f us/frame |",
	                                 (unsigned long long)frames, double(total_draws) / double(frames), double(total_ns) / 1000.0 / double(frames)));
	for (const Row& r: rows) {
		if (r.draws == 0 && r.ns == 0) continue;
		if (at >= sizeof(line)) break;
		at += size_t(std::snprintf(line + at, sizeof(line) - at, " %s: %.0f draws/f %.0f us/f %.2f us/draw (%.0f entradas/f) |",
		                           SlotName(r.i), double(r.draws) / double(frames),
		                           double(r.ns) / 1000.0 / double(frames), r.draws ? double(r.ns) / 1000.0 / double(r.draws) : 0.0,
		                           double(r.entries) / double(frames)));
	}
	std::printf("%s ultimo_sitio_sin_nombre=eboot+0x%llx\n", line, (unsigned long long)g_last_other.load(std::memory_order_relaxed));
	std::fflush(stdout);
}

// Hilo de la GPU, en CommandProcessor::Flip.
inline void OnFlip() {
	static uint64_t frames = 0;
	Report(++frames);
}

} // namespace DsCensoEscena
