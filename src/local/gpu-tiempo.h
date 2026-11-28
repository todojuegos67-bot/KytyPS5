#pragma once
// (gpu-tiempo) Con KYTY_TECHO=1 (y KYTY_GPU_TIEMPO distinto de "0"): tiempo REAL de la GPU por fotograma con marcas de
// tiempo de Vulkan. Solo mide: no cambia el orden de envio ni lo que se dibuja. Apagado: una prueba de un bool estatico.
//
//  - Pool de consultas propio, NSlots fotogramas en vuelo (anillo); cada buffer de comandos del planificador
//    (BeginCommand = el principal, UploadPrologue = prologo de subidas) recibe TOP_OF_PIPE al empezar y
//    BOTTOM_OF_PIPE en SubmitBuffers antes de cerrarse (por el mismo despachador: con la grabacion diferida la marca
//    va en orden en el flujo). Dentro del principal, una marca en cada trozo de submission (GuestGpu::Process) con
//    el grupo del sitio del juego (DsCensoDcb) y otra al acabar el trozo (grupo "emulador").
//  - Se lee con retraso (Lag fotogramas) y sin esperar (WITH_AVAILABILITY); si un fotograma aun no esta listo se
//    deja para el siguiente flip y su hueco del anillo no se reutiliza (marcas de ese fotograma: "sin_hueco").
//  - Linea "[gpu-tiempo]" cada 5 s: GPU ocupada ms/fot (union de intervalos de los buffers: media p50 p95),
//    ociosa = fotograma - ocupada, abarcado (primera marca a ultima), ms/fot por grupo, y la relacion con el render
//    ocupado del hilo (fotograma - libre, el mismo calculo que [techo]).
// Ghidra (projelf_par1), sitios = quien llama a RenderThread::KickoffSubmitJob 0xa08ae0 (xrefs):
//   FUN_00a090e0 escena diferida (0xa09138/0xa09375/0xa09a76), FUN_00a0eb80 forward (0xa0ed39/0xa0ed89/0xa0ee72; las
//   sombras CascadedShadows 0xa11220 se llaman en 0xa0ee11, entre el #2 y el #3 => van en la DCB "forward #3"),
//   FUN_00a12350 (RenderDepth/GBuffer/Decal, 3 envios), FUN_00a14886 bucle (BinkManager::UpdateJob, 0xa150f5),
//   FUN_00a0de50 pre-frame, FUN_00a16a20 UpdateTextureStreamingGPUData.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "local/techo.h"

namespace GpuTiempo {

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* g = std::getenv("KYTY_GPU_TIEMPO");
		return Techo::On() && !(g != nullptr && g[0] == '0' && g[1] == '\0');
	}();
	return on;
}

enum Group : uint8_t { GEscena, GFwd12, GFwd3Sombras, GA12350, GBucle, GPreFrame, GAcb, GOtroSitio, GEmu, Groups };
inline const char* const kGroupName[Groups] {"escena a090e0", "forward a0eb80 #1-2", "forward #3+sombras", "a12350",
                                             "bucle a14886",  "pre-frame a0de50",    "ACB",                "otro sitio",
                                             "emulador"};
// Grupo por direccion de retorno del sitio (DsCensoDcb kNames).
[[nodiscard]] inline uint8_t GroupOf(uint32_t ret, bool acb) {
	if (acb) return GAcb;
	switch (ret) {
	case 0xa0913d: case 0xa0937a: case 0xa09a7b: return GEscena;
	case 0xa0ed3e: case 0xa0ed8e: return GFwd12;
	case 0xa0ee77: return GFwd3Sombras;
	case 0xa12393: case 0xa12597: case 0xa12707: return GA12350;
	case 0xa150fa: return GBucle;
	case 0xa0decd: return GPreFrame;
	default: return GOtroSitio;
	}
}

constexpr uint32_t NSlots = 8, Lag = 4, Records = 256, PerRecord = 64, SlotQueries = Records * PerRecord;
constexpr uint32_t TotalQueries = NSlots * SlotQueries;

struct Record {
	VkCommandBuffer cmd = nullptr;
	uint32_t        n   = 0;     // consultas usadas (desde Base)
	uint8_t         group[PerRecord] {};
	std::atomic<uint32_t> state {0}; // 0 libre, 1 abierto, 2 enviado, 3 perdido
};
struct Slot {
	std::atomic<uint32_t> next {0};
	std::atomic<uint32_t> busy {0}; // aun por leer
	uint64_t              frame = 0;
	Record                rec[Records];
};
struct State {
	VkDevice                 device = nullptr;
	VkQueryPool              pool   = VK_NULL_HANDLE;
	PFN_vkCmdWriteTimestamp  write  = nullptr;
	PFN_vkGetQueryPoolResults get   = nullptr;
	PFN_vkResetQueryPool     reset  = nullptr;
	double                   period = 0;
	uint32_t                 bits   = 0;
	uint64_t                 mask   = 0;
	std::atomic<bool>        ok {false};
	std::atomic<uint64_t>    frame {0};
	Slot                     slots[NSlots];
	std::atomic<uint64_t>    no_slot {0}, no_rec {0}, full {0}, lost {0}, late {0}, cost_ns {0};
	// ventana (hilo del flip)
	std::vector<double> busy_ms;
	double sum_span = 0, sum_group[Groups] {};
	uint64_t frames_read = 0;
};
inline State* g_s = nullptr;

inline thread_local Record* t_open[4] {};
inline thread_local Record* t_main = nullptr;
inline thread_local uint8_t t_group = GEmu;


// Una vez, desde el planificador (hilo que abre el primer buffer). Los PFN vienen del despachador por defecto (el
// vkCmdWriteTimestamp ya redirigido a la grabacion diferida si esta activa).
inline void Init(VkDevice device, VkPhysicalDevice phys, uint32_t family, float period_ns, PFN_vkCmdWriteTimestamp write,
                 PFN_vkGetQueryPoolResults get, PFN_vkResetQueryPool reset, PFN_vkCreateQueryPool create,
                 PFN_vkGetPhysicalDeviceQueueFamilyProperties families) {
	auto* s   = new State;
	s->device = device, s->write = write, s->get = get, s->reset = reset, s->period = period_ns;
	uint32_t n = 0;
	families(phys, &n, nullptr);
	std::vector<VkQueueFamilyProperties> f(n);
	families(phys, &n, f.data());
	s->bits = family < n ? f[family].timestampValidBits : 0;
	std::printf("[gpu-tiempo] timestampValidBits=%u timestampPeriod=%.3f ns (cola %u)\n", s->bits, period_ns, family);
	if (s->bits == 0 || period_ns <= 0 || write == nullptr || get == nullptr || reset == nullptr) {
		std::printf("[gpu-tiempo] la cola no admite marcas de tiempo: sin tiempo de GPU\n");
		g_s = s;
		return;
	}
	s->mask = s->bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << s->bits) - 1u;
	VkQueryPoolCreateInfo info {};
	info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	info.queryCount = TotalQueries;
	if (create(device, &info, nullptr, &s->pool) != VK_SUCCESS) {
		std::printf("[gpu-tiempo] no se pudo crear el pool: sin tiempo de GPU\n");
		g_s = s;
		return;
	}
	reset(device, s->pool, 0, TotalQueries);
	if (s->bits < 40)
		std::printf("[gpu-tiempo] aviso: %u bits validos, el contador da la vuelta cada %.1f s (se corrige dentro del fotograma)\n",
		            s->bits, static_cast<double>(s->mask) * period_ns / 1e9);
	std::fflush(stdout);
	s->ok.store(true, std::memory_order_release);
	g_s = s;
}

inline void Stamp(Record& r, uint32_t slot_index, uint32_t rec_index, uint8_t group, bool top) {
	const uint32_t q = slot_index * SlotQueries + rec_index * PerRecord + r.n;
	g_s->write(r.cmd, top ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_s->pool, q);
	r.group[r.n++] = group;
}
inline void Locate(const Record* r, uint32_t* slot, uint32_t* idx) {
	for (uint32_t s = 0; s < NSlots; ++s) {
		const Record* b = g_s->slots[s].rec;
		if (r >= b && r < b + Records) {
			*slot = s, *idx = static_cast<uint32_t>(r - b);
			return;
		}
	}
}

// Un buffer del planificador empieza (`main`: el abierto principal; si no, un prologo de subidas).
inline void Begin(void* raw, bool main) {
	if (g_s == nullptr || !g_s->ok.load(std::memory_order_relaxed) || raw == nullptr) return;
	const uint64_t frame = g_s->frame.load(std::memory_order_relaxed);
	const uint32_t si    = static_cast<uint32_t>(frame % NSlots);
	Slot&          slot  = g_s->slots[si];
	if (slot.busy.load(std::memory_order_acquire) != 1 || slot.frame != frame) {
		g_s->no_slot.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const uint32_t i = slot.next.fetch_add(1, std::memory_order_relaxed);
	if (i >= Records) {
		g_s->no_rec.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	Record& r = slot.rec[i];
	r.cmd     = static_cast<VkCommandBuffer>(raw);
	r.n       = 0;
	r.state.store(1, std::memory_order_relaxed);
	Stamp(r, si, i, main ? t_group : static_cast<uint8_t>(GEmu), true);
	Record** free_slot = &t_open[0];
	for (auto& o: t_open)
		if (o == nullptr) {
			free_slot = &o;
			break;
		}
	if (*free_slot != nullptr) (*free_slot)->state.store(3, std::memory_order_relaxed); // (nunca enviado)
	*free_slot = &r;
	if (main) t_main = &r;
}

// Cambio de grupo dentro del buffer principal abierto de este hilo (inicio/fin de un trozo de submission).
inline void Mark(uint8_t group) {
	t_group = group;
	Record* r = t_main;
	if (r == nullptr || r->n == 0 || r->group[r->n - 1] == group) return;
	if (r->n + 2 > PerRecord) {
		g_s->full.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	uint32_t si = 0, idx = 0;
	Locate(r, &si, &idx);
	Stamp(*r, si, idx, group, false);
}

// SubmitBuffers, antes de cerrar los buffers: la marca de fin. `ended`: cerrado en otro hilo (sin marca: perdido).
inline void Submit(void* raw, bool ended) {
	if (g_s == nullptr || raw == nullptr) return;
	for (auto& o: t_open) {
		if (o == nullptr || o->cmd != static_cast<VkCommandBuffer>(raw)) continue;
		Record* r = o;
		o         = nullptr;
		if (t_main == r) t_main = nullptr;
		if (ended) {
			r->state.store(3, std::memory_order_relaxed);
			return;
		}
		uint32_t si = 0, idx = 0;
		Locate(r, &si, &idx);
		Stamp(*r, si, idx, r->group[r->n - 1], false);
		r->state.store(2, std::memory_order_release);
		return;
	}
}

// Trozo de submission (GuestGpu::Process): marca al empezar con el grupo del sitio y al acabar "emulador".
struct SliceScope {
	bool on;
	SliceScope(bool active, uint32_t site_ret, bool acb) : on(active && On()) {
		if (on) Mark(GroupOf(site_ret, acb));
	}
	~SliceScope() {
		if (on) Mark(GEmu);
	}
	SliceScope(const SliceScope&)            = delete;
	SliceScope& operator=(const SliceScope&) = delete;
};

// Lee un fotograma ya enviado; false si alguna marca no esta lista aun.
inline bool ReadSlot(uint32_t si) {
	Slot&          slot = g_s->slots[si];
	const uint32_t n    = std::min(slot.next.load(std::memory_order_acquire), Records);
	struct Span { uint64_t b, e; };
	std::vector<Span> spans;
	spans.reserve(n);
	double group_ticks[Groups] {};
	uint64_t pairs[PerRecord * 2];
	for (uint32_t i = 0; i < n; ++i) {
		Record& r = slot.rec[i];
		if (r.state.load(std::memory_order_acquire) != 2 || r.n < 2) continue;
		const VkResult res = g_s->get(g_s->device, g_s->pool, si * SlotQueries + i * PerRecord, r.n, sizeof(uint64_t) * 2 * r.n,
		                              pairs, sizeof(uint64_t) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
		if (res != VK_SUCCESS && res != VK_NOT_READY) continue;
		for (uint32_t k = 0; k < r.n; ++k)
			if (pairs[k * 2 + 1] == 0) return false;
		// Relativo a la primera marca del buffer (corrige la vuelta del contador dentro del buffer).
		const uint64_t b = pairs[0] & g_s->mask;
		uint64_t       prev = 0;
		for (uint32_t k = 1; k < r.n; ++k) {
			uint64_t d = ((pairs[k * 2] & g_s->mask) - b) & g_s->mask;
			d          = std::max(d, prev);
			group_ticks[r.group[k - 1]] += static_cast<double>(d - prev);
			prev = d;
		}
		spans.push_back({b, b + prev});
	}
	// Union de intervalos (con la vuelta del contador: relativo a la primera marca del fotograma).
	double busy = 0, span = 0;
	if (!spans.empty()) {
		const uint64_t base = spans[0].b;
		for (auto& sp: spans) {
			const uint64_t len = sp.e - sp.b;
			uint64_t       rel = (sp.b - base) & g_s->mask;
			if (rel > g_s->mask / 2) rel = 0; // (anterior a la base: muy raro; se pega a la base)
			sp.b = rel, sp.e = rel + len;
		}
		std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& c) { return a.b < c.b; });
		uint64_t cb = spans[0].b, ce = spans[0].e, lo = spans[0].b, hi = 0, total = 0;
		for (const auto& sp: spans) {
			hi = std::max(hi, sp.e);
			if (sp.b > ce) {
				total += ce - cb;
				cb = sp.b, ce = sp.e;
			} else ce = std::max(ce, sp.e);
		}
		total += ce - cb;
		busy = static_cast<double>(total) * g_s->period / 1e6;
		span = static_cast<double>(hi - lo) * g_s->period / 1e6;
	}
	g_s->busy_ms.push_back(busy);
	g_s->sum_span += span;
	for (uint32_t k = 0; k < Groups; ++k) g_s->sum_group[k] += group_ticks[k] * g_s->period / 1e6;
	++g_s->frames_read;
	return true;
}
inline void ReleaseSlot(uint32_t si) {
	Slot& slot = g_s->slots[si];
	const uint32_t n = std::min(slot.next.load(std::memory_order_acquire), Records);
	for (uint32_t i = 0; i < n; ++i)
		if (slot.rec[i].state.load(std::memory_order_relaxed) == 3) g_s->lost.fetch_add(1, std::memory_order_relaxed);
	g_s->reset(g_s->device, g_s->pool, si * SlotQueries, SlotQueries);
	for (auto& r: slot.rec) r.state.store(0, std::memory_order_relaxed), r.n = 0;
	slot.next.store(0, std::memory_order_relaxed);
	slot.busy.store(0, std::memory_order_release);
}

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// LiveControl::Flip (hilo de render): cierra el fotograma, lee los de hace >= Lag sin esperar, linea cada 5 s.
inline void Flip(int64_t flip_ns, int64_t idle_ns_total) {
	if (!On() || g_s == nullptr || !g_s->ok.load(std::memory_order_acquire)) return;
	auto&          s  = *g_s;
	const int64_t  t0 = NowNs();
	static uint64_t read_next = 1; // (el fotograma 0 abre antes del primer flip: se abre aqui el 1)
	static int64_t  last_t = 0, last_idle = 0;
	static uint64_t frames = 0;
	// Abre el fotograma siguiente si su hueco esta libre.
	const uint64_t f  = s.frame.load(std::memory_order_relaxed) + 1;
	Slot&          ns = s.slots[f % NSlots];
	if (ns.busy.load(std::memory_order_acquire) == 0) {
		ns.frame = f;
		ns.busy.store(1, std::memory_order_release);
	}
	s.frame.store(f, std::memory_order_release);
	// Lee los fotogramas cerrados hace Lag o mas.
	while (read_next + Lag <= f) {
		Slot& rs = s.slots[read_next % NSlots];
		if (rs.busy.load(std::memory_order_acquire) == 1 && rs.frame == read_next) {
			if (!ReadSlot(static_cast<uint32_t>(read_next % NSlots))) {
				if (read_next + 120 > f) break; // (aun en la GPU: se reintenta en el siguiente flip, sin esperar)
				s.late.fetch_add(1, std::memory_order_relaxed);
			}
			ReleaseSlot(static_cast<uint32_t>(read_next % NSlots));
		}
		++read_next;
	}
	++frames;
	s.cost_ns.fetch_add(static_cast<uint64_t>(NowNs() - t0), std::memory_order_relaxed);
	if (last_t == 0) {
		last_t = flip_ns, last_idle = idle_ns_total, frames = 0;
		return;
	}
	if (flip_ns - last_t < 5'000'000'000ll || frames == 0 || s.frames_read == 0) return;
	const double n       = static_cast<double>(frames);
	const double frame   = static_cast<double>(flip_ns - last_t) / 1e6 / n;
	const double render  = frame - static_cast<double>(idle_ns_total - last_idle) / 1e6 / n; // (= render ocupado de [techo])
	auto&        v       = s.busy_ms;
	const double m       = static_cast<double>(s.frames_read);
	double       sum     = 0;
	for (double x: v) sum += x;
	std::sort(v.begin(), v.end());
	const double mean = sum / m, p50 = v[v.size() / 2], p95 = v[std::min(v.size() - 1, v.size() * 95 / 100)];
	char groups[512];
	int  at = 0;
	for (uint32_t k = 0; k < Groups && at < static_cast<int>(sizeof(groups)) - 64; ++k)
		if (s.sum_group[k] / m >= 0.005)
			at += std::snprintf(groups + at, sizeof(groups) - at, " %s %.2f;", kGroupName[k], s.sum_group[k] / m);
	groups[at] = '\0';
	std::printf("[gpu-tiempo] fot %.2f ms (%.0f fot, %.0f leidos) | GPU ocupada media %.2f p50 %.2f p95 %.2f ms/fot | GPU ociosa "
	            "%.2f ms/fot | abarcado %.2f | render ocupado %.2f -> GPU/render %.2f, holgura GPU %.2f ms | por grupo (ms/fot "
	            "GPU):%s | sin_hueco %llu sin_registro %llu lleno %llu perdidos %llu tarde %llu | coste %.3f ms/fot\n",
	            frame, n, m, mean, p50, p95, std::max(0.0, frame - mean), s.sum_span / m, render, render > 0 ? mean / render : 0.0,
	            render - mean, groups, (unsigned long long)s.no_slot.exchange(0), (unsigned long long)s.no_rec.exchange(0),
	            (unsigned long long)s.full.exchange(0), (unsigned long long)s.lost.exchange(0),
	            (unsigned long long)s.late.exchange(0), static_cast<double>(s.cost_ns.exchange(0)) / 1e6 / n);
	std::fflush(stdout);
	v.clear();
	s.sum_span = 0;
	for (double& g: s.sum_group) g = 0;
	s.frames_read = 0;
	last_t = flip_ns, last_idle = idle_ns_total, frames = 0;
}

} // namespace GpuTiempo
