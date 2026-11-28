#pragma once
// (tej) KYTY_TEJ, tejido propio FASE 1 "sombra" (tools/plan-weave-propio.md §5). Apagado por defecto. SOLO MIDE: la imagen
// no puede cambiar (nada de lo que hace el trabajador se envia, se cose ni se sube a recursos del juego).
//   0 apagado (identico a antes: un static por gancho).
//   1 sombra encendida.
//   2 A/B cada ventana de 5 s de [techo] (KYTY_TECHO=1): B = sombra encendida; delta interpolado B-A del render ocupado
//     (objetivo 0 +-0.2 ms/fot: lo que la sombra le cuesta al juego).
// Que hace:
// - Un solo hilo (Kyty.Tej, lector + trabajador), prioridad por debajo de la normal (por debajo del render), afinidad a
//   KYTY_RENDER_CPUS (o a la L3 del hilo de render) SIN el nucleo del hilo de render (ni su hermano SMT; se rehace cuando
//   el render cambia de nucleo). Lo despierta cada envio de la cola 0 (GuestGpu::Enqueue: ahi se COPIAN las ordenes de
//   la DCB, la memoria del juego no se lee despues). Solo traduce si en la ventana anterior (100 ms) hubo >= 1 nucleo
//   libre en la maquina (CPU ociosa de los CPUs del proceso); si no, no toma nada (coste 0 para el juego).
// - Adelanta registros: su propio CommandProcessor encadenado DCB a DCB (el estado que deja una es el inicio de la
//   siguiente); si la cadena se rompe pide al hilo de render el estado tras la DCB anterior (una copia, solo entonces).
// - Traduce la DCB con la ruta normal en modo especulativo (Speculation::Shadow = Translate de KYTY_SPECULATE: estado
//   por hilo, Spec::State rechaza todo lo que escribiria estado compartido: registros de buffers nuevos, pares, sets,
//   subidas, transiciones -> "hueco" contado; bc_par: lookup concurrente del indice de BufferCache con cerrojo
//   compartido, que KYTY_TEJ enciende) a un recorder propio que NUNCA se envia (Finish: descartado al acabar) y a
//   anillos de subida propios (16 + 8 MiB) que nadie lee. Lo que el hilo de render libera mientras lee: lo protege el
//   punto de espera que ya existe (Spec::EnterPacket/LeavePackets: buffers e imagenes retirados se destruyen cuando el
//   paquete paso) y el catalogo se lee bajo su cerrojo compartido; el barrido de pipelines de mundos dejados se salta
//   con KYTY_TEJ (como con KYTY_SPECULATE).
// - Corta paquetes de 64 draws: al primer draw de cada uno guarda la foto del procesador y las epocas (buffers
//   registrados, imagenes, mapeos) y lo publica al acabarlo.
// - El hilo de render NO cambia su camino: en el primer draw de cada paquete (misma posicion en la DCB) compara su
//   estado con la foto (SameState) y las epocas, y cuenta si el paquete se habria podido coser y si llego a tiempo.
// Linea [tej] cada 5 s (con KYTY_TECHO=1).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/speculation.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "local/local-platform.h"
#include "local/speculation-state.h"
#include "local/techo.h"

namespace Tej {

using Libs::Graphics::CommandProcessor;
using Libs::Graphics::RenderContext;
using Libs::Graphics::Speculation;

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_TEJ");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m != 1 && m != 2) m = 0;
		if (m == 2 && !Techo::On()) {
			std::printf("[tej] KYTY_TEJ=2 sin KYTY_TECHO=1: modo 1 (sombra siempre, sin linea)\n");
			m = 1;
		}
		if (m != 0)
			std::printf("[tej] KYTY_TEJ=%d (%s): F1 sombra, SOLO MIDE (nada se cose ni se envia)\n", m,
			            m == 1 ? "sombra encendida" : "A/B cada 5 s, B = sombra encendida");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false};
[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || (m == 2 && g_ab_on.load(std::memory_order_relaxed));
}

inline constexpr uint32_t PacketDraws = 64, MaxPackets = 16, Slots = 8, MaxAhead = 6, MaxQueued = 16;

// Contadores (relaxed; los del hilo de render por paquete o por DCB, no por draw).
enum C : uint32_t {
	// hilo de render
	GDraws,     // draws de las DCB de la cola 0 en ventanas encendidas
	GDcbs,      // DCB de graficos de la cola 0
	GDcbSlot,   // de ellas, con traduccion de la sombra
	GPkOk,      // paquetes validados a tiempo (registros y epocas iguales)
	GDrawsOk,   // draws cosibles (de paquetes validados, sin los huecos)
	GHolesOk,   // huecos (draws) dentro de paquetes validados
	GPkRegs,    // paquete a tiempo, registros distintos
	GDrawsRegsOk, // draws de paquetes con registros iguales (cota sin epocas)
	GPkEpBuf,   // registros iguales, epoca de buffers movida
	GPkEpImg,   // ... de imagenes
	GPkEpMap,   // ... de mapeos
	GPkLate,    // publicado despues de que el render pasara
	GPkSkip,    // el render no paso por esa posicion (lote / otro camino)
	GTailLate,  // DCB acabada por el render antes que por la sombra
	GValTicks,  // ticks TSC de validacion
	GValN,      // validaciones
	GPublish,   // estados publicados para reenganchar la cadena
	// trabajador
	WJobs, WNs, WDraws, WDropFull, WBehind, WNoCpu, WNoState, WOther, WStops, WFailed, WHolesDraw, WHolesOther, WOver,
	WRepins, WAheadWaitNs, WCopyBytes,
	NCount
};
inline std::atomic<uint64_t> g_c[NCount] {};
inline void Add(C i, uint64_t v = 1) { g_c[i].fetch_add(v, std::memory_order_relaxed); }

// Hilo de render: el ultimo serial de la cola 0 que empezo y su CPU.
inline std::atomic<uint64_t> g_gpu_serial {0};
inline std::atomic<uint32_t> g_render_cpu {0xffffffffu};
// CPU libre (nucleos, x100) en la ultima muestra; suma y muestras para la linea.
inline std::atomic<int64_t>  g_free_x100 {-100};
inline std::atomic<uint64_t> g_free_sum {0}, g_free_n {0};

struct Packet {
	uint64_t                          key = 0;
	uint32_t                          draws = 0, holes = 0;
	uint64_t                          ep_buf = 0, ep_img = 0, ep_map = 0;
	uint64_t                          publish_tsc = 0;
	std::unique_ptr<CommandProcessor> cp; // (foto al primer draw; del trabajador, creada en su hilo)
};
struct Slot {
	std::atomic<uint64_t> serial {0};
	std::atomic<uint32_t> count {0}; // paquetes publicados (inmutables desde entonces)
	std::atomic<bool>     done {false};
	Packet                pk[MaxPackets];
};

inline thread_local int t_side = 0; // 1 hilo de render (en una DCB con sombra encendida), 2 trabajador

[[nodiscard]] inline bool IsDrawHeader(uint32_t header) {
	switch ((header >> 8u) & 0xffu) {
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_2:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_AUTO:
		case Libs::Graphics::Pm4::IT_DRAW_INDIRECT:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_INDIRECT:
		case Libs::Graphics::Pm4::IT_DRAW_INDIRECT_MULTI:
		case Libs::Graphics::Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

class Worker {
public:
	struct Job {
		enum Kind : uint8_t { Graphics, Flip, Other };
		uint64_t              serial = 0;
		Kind                  kind   = Other;
		bool                  reset  = false;
		std::vector<uint32_t> cmds;
	};

	explicit Worker(RenderContext& r)
	    : m_r(r), m_spec(r, Speculation::ShadowTag {}), m_chain(std::make_unique<CommandProcessor>(r, 0)),
	      m_end(std::make_unique<CommandProcessor>(r, 0)), m_published(std::make_unique<CommandProcessor>(r, 0)) {
		m_thread = std::thread([this] { Run(); });
	}
	~Worker() {
		{
			std::lock_guard g(m_mutex);
			m_quit = true;
		}
		m_cv.notify_all();
		if (m_thread.joinable()) m_thread.join();
	}
	Worker(const Worker&)            = delete;
	Worker& operator=(const Worker&) = delete;

	// Hilo del juego (GuestGpu::Enqueue, bajo su cerrojo de colas): la orden copiada.
	void Push(uint64_t serial, Job::Kind kind, bool reset, std::span<const uint32_t> commands) {
		{
			std::lock_guard g(m_mutex);
			if (m_jobs.size() >= MaxQueued) {
				Add(WDropFull); // (la cadena se rompe sola: el serial siguiente no sigue al ultimo traducido)
				return;
			}
			Job job;
			if (!m_pool.empty()) job.cmds = std::move(m_pool.back()), m_pool.pop_back();
			job.serial = serial;
			job.kind   = kind;
			job.reset  = reset;
			if (kind == Job::Graphics) {
				job.cmds.assign(commands.begin(), commands.end());
				Add(WCopyBytes, commands.size() * 4u);
			} else {
				job.cmds.clear();
			}
			m_jobs.push_back(std::move(job));
		}
		m_cv.notify_one();
	}

	// Hilo de render, al acabar un envio de la cola 0: si la sombra espera el estado tras ese serial, una copia.
	void GpuDone(uint64_t serial, const CommandProcessor& cp) {
		if (m_want_serial.load(std::memory_order_acquire) != serial) return;
		std::lock_guard g(m_pub_mutex);
		m_published->CopyStateFrom(cp, false);
		m_pub_serial = serial;
		m_want_serial.store(0, std::memory_order_relaxed);
		Add(GPublish);
	}

	[[nodiscard]] Slot& SlotOf(uint64_t serial) { return m_slots[serial % Slots]; }

	// Trabajador: un draw de su traduccion (tope del bucle PM4, antes de traducirlo).
	void WorkerDraw(const CommandProcessor& cp, RenderContext& r, uint64_t key) {
		if (m_slot == nullptr || key == m_last_key) return; // (el mismo paquete traducido otra vez: corte de bc_par)
		m_last_key     = key;
		const auto ord = m_ord++;
		if (ord % PacketDraws == 0) {
			Close();
			const uint32_t j = ord / PacketDraws;
			if (j >= MaxPackets) {
				Add(WOver);
				return;
			}
			auto& p = m_slot->pk[j];
			if (!p.cp) p.cp = std::make_unique<CommandProcessor>(m_r, 0);
			p.cp->CopyStateFrom(cp, false);
			p.key    = key;
			p.draws  = 0;
			p.holes  = 0;
			p.ep_buf = r.GetBufferCache().RegistrationEpoch();
			p.ep_img = r.GetTextureCache().ResolutionEpoch();
			p.ep_map = r.GetGpuResources().MappingEpoch();
			m_open   = static_cast<int>(j);
		}
		if (m_open >= 0) ++m_slot->pk[m_open].draws;
	}

	static void HoleHook(const uint32_t* packet, const char* reason);
	void        Hole(const uint32_t* packet, const char* reason) {
        const bool draw = packet != nullptr && IsDrawHeader(*packet);
        if (draw && m_open >= 0) ++m_slot->pk[m_open].holes;
        Add(draw ? WHolesDraw : WHolesOther);
        if (draw) ++m_job_holes[reason != nullptr ? reason : "?"];
	}

	// Hilo del informe.
	void TakeReasons(std::vector<std::pair<const char*, uint64_t>>& holes, std::vector<std::pair<const char*, uint64_t>>& stops) {
		std::lock_guard g(m_reason_mutex);
		holes.assign(m_holes.begin(), m_holes.end());
		stops.assign(m_stops.begin(), m_stops.end());
		m_holes.clear();
		m_stops.clear();
	}

private:
	void Close() {
		if (m_slot == nullptr || m_open < 0) return;
		m_slot->pk[m_open].publish_tsc = __rdtsc();
		m_slot->count.store(static_cast<uint32_t>(m_open) + 1, std::memory_order_release);
		m_open = -1;
	}

	void SampleCpu() {
		const auto now = std::chrono::steady_clock::now();
		if (now - m_sample_at < std::chrono::milliseconds(100)) return;
		uint64_t idle = 0, mask = 0;
		if (!LocalPlatform::ProcessCpusIdleTime(&idle, &mask)) {
			m_gate = true; // (sin dato: no se limita)
			g_free_x100.store(-100, std::memory_order_relaxed);
			m_sample_at = now;
			return;
		}
		if (m_last_idle != 0 && idle >= m_last_idle) {
			const double wall_100ns = std::chrono::duration<double>(now - m_sample_at).count() * 1e7;
			const double free       = wall_100ns > 0 ? static_cast<double>(idle - m_last_idle) / wall_100ns : 0;
			m_gate                  = free >= 1.0;
			g_free_x100.store(static_cast<int64_t>(free * 100), std::memory_order_relaxed);
			g_free_sum.fetch_add(static_cast<uint64_t>(free * 100), std::memory_order_relaxed);
			g_free_n.fetch_add(1, std::memory_order_relaxed);
		}
		m_last_idle = idle;
		m_sample_at = now;
	}

	// Afinidad: la CCD/L3 del render (KYTY_RENDER_CPUS si existe) sin el nucleo del render (ni su hermano SMT).
	void Pin() {
		const uint32_t rc = g_render_cpu.load(std::memory_order_relaxed);
		if (rc >= 64 || rc == m_pinned_for) return;
		m_pinned_for = rc;
		uint64_t base = LocalPlatform::CpuListToMask(std::getenv("KYTY_RENDER_CPUS"));
		if (base == 0) base = LocalPlatform::L3MaskOfCpu(rc);
		uint64_t core = LocalPlatform::CoreMaskOfCpu(rc);
		if (core == 0) core = uint64_t {1} << rc;
		const uint64_t m = base & ~core;
		if (m != 0 && LocalPlatform::SetCurrentThreadMask(m)) Add(WRepins);
	}

	void Run() {
		LocalPlatform::SetThreadName("Kyty.Tej");
		LocalPlatform::MakeBackgroundThread(nullptr); // (THREAD_PRIORITY_BELOW_NORMAL: por debajo del render)
		m_spec.ShadowThreadInit();
		Libs::Graphics::Spec::t_hole_hook = &Worker::HoleHook;
		t_side                            = 2;
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return m_quit || !m_jobs.empty(); });
			if (m_quit) break;
			lock.unlock();
			SampleCpu();
			lock.lock();
			if (m_jobs.empty()) continue;
			Job job = std::move(m_jobs.front());
			m_jobs.pop_front();
			lock.unlock();
			Handle(job);
			lock.lock();
			if (m_pool.size() < MaxQueued) m_pool.push_back(std::move(job.cmds));
		}
		t_side = 0;
	}

	void Handle(Job& job) {
		// (La cadena sigue solo con serial consecutivo: un envio perdido la rompe.)
		const bool consecutive = m_chain_ok && m_chain_serial + 1 == job.serial;
		if (job.kind == Job::Flip) { // (entre DCB el estado del procesador sigue: Speculation::Job::Kind::Flip)
			if (consecutive) m_chain_serial = job.serial;
			else m_chain_ok = false;
			return;
		}
		if (job.kind == Job::Other) {
			m_chain_ok = false;
			Add(WOther);
			return;
		}
		uint64_t gpu = g_gpu_serial.load(std::memory_order_acquire);
		if (job.serial <= gpu) { // (el render ya la empezo: llegaria tarde entera)
			m_chain_ok = false;
			Add(WBehind);
			return;
		}
		if (job.serial > gpu + MaxAhead) { // (no pisar una ranura que el render aun lee)
			const auto t0 = Speculation::NowNs();
			while (job.serial > gpu + MaxAhead && !Quit()) {
				std::this_thread::sleep_for(std::chrono::microseconds(200));
				gpu = g_gpu_serial.load(std::memory_order_acquire);
			}
			Add(WAheadWaitNs, Speculation::NowNs() - t0);
			if (Quit()) return;
		}
		if (!m_gate) {
			m_chain_ok = false;
			Add(WNoCpu);
			return;
		}
		if (!consecutive) {
			// Estado tras la anterior: lo pide al render (una copia al acabarla) y espera, sin pasar a la siguiente.
			m_want_serial.store(job.serial - 1, std::memory_order_release);
			bool got = false;
			for (;;) {
				{
					std::lock_guard g(m_pub_mutex);
					if (m_pub_serial == job.serial - 1) {
						m_chain->CopyStateFrom(*m_published, false);
						got = true;
					}
				}
				if (got || Quit() || g_gpu_serial.load(std::memory_order_acquire) >= job.serial) break;
				std::this_thread::sleep_for(std::chrono::microseconds(100));
			}
			if (!got) {
				m_want_serial.store(0, std::memory_order_relaxed);
				Add(WNoState);
				return;
			}
		}
		Pin();
		if (job.reset) m_chain->Reset();
		m_chain->ResetDeCe();
		auto& slot = SlotOf(job.serial);
		slot.serial.store(0, std::memory_order_release);
		slot.count.store(0, std::memory_order_release);
		slot.done.store(false, std::memory_order_release);
		slot.serial.store(job.serial, std::memory_order_release);
		m_slot     = &slot;
		m_open     = -1;
		m_ord      = 0;
		m_last_key = ~uint64_t {0};
		m_job_holes.clear();
		const auto t0  = Speculation::NowNs();
		const auto out = m_spec.Shadow(*m_chain, std::span<const uint32_t>(job.cmds.data(), job.cmds.size()), *m_end);
		const auto t1  = Speculation::NowNs();
		Close();
		slot.done.store(true, std::memory_order_release);
		m_slot = nullptr;
		m_chain->CopyStateFrom(*m_end, false);
		m_chain_serial = job.serial;
		m_chain_ok     = !out.failed && out.stop == nullptr;
		Add(WJobs);
		Add(WNs, t1 - t0);
		Add(WDraws, m_ord);
		if (out.failed) Add(WFailed);
		if (out.stop != nullptr) Add(WStops);
		std::lock_guard g(m_reason_mutex);
		for (const auto& [why, n]: m_job_holes) m_holes[why] += n;
		if (out.stop != nullptr) ++m_stops[out.stop];
	}

	[[nodiscard]] bool Quit() {
		std::lock_guard g(m_mutex);
		return m_quit;
	}

	RenderContext&                    m_r;
	Speculation                       m_spec;
	std::unique_ptr<CommandProcessor> m_chain, m_end, m_published;
	std::thread                       m_thread;
	std::mutex                        m_mutex;
	std::condition_variable           m_cv;
	std::deque<Job>                   m_jobs;
	std::vector<std::vector<uint32_t>> m_pool;
	bool                              m_quit = false;
	// (trabajador)
	bool                                       m_chain_ok = false, m_gate = true;
	uint64_t                                   m_chain_serial = 0, m_last_idle = 0, m_last_key = 0;
	uint32_t                                   m_pinned_for = 0xffffffffu, m_ord = 0;
	int                                        m_open = -1;
	Slot*                                      m_slot = nullptr;
	std::chrono::steady_clock::time_point      m_sample_at {};
	std::unordered_map<const char*, uint64_t>  m_job_holes;
	// (estado publicado por el render para reenganchar)
	std::mutex            m_pub_mutex;
	uint64_t              m_pub_serial = 0;
	std::atomic<uint64_t> m_want_serial {0};
	// (motivos, para el informe)
	std::mutex                                m_reason_mutex;
	std::unordered_map<const char*, uint64_t> m_holes, m_stops;
	Slot                                      m_slots[Slots];
};

inline std::atomic<Worker*> g_worker {nullptr};

inline void Worker::HoleHook(const uint32_t* packet, const char* reason) {
	if (auto* w = g_worker.load(std::memory_order_relaxed); w != nullptr && t_side == 2) w->Hole(packet, reason);
}

inline void Window(double busy_ms, double fps, double frames, bool keep);

// Hilo de render, al arrancar / al parar.
inline void Start(RenderContext& r) {
	if (Mode() == 0 || g_worker.load(std::memory_order_relaxed) != nullptr) return;
	g_worker.store(new Worker(r), std::memory_order_release);
	Techo::g_tej_window = &Window;
}
inline void Stop() {
	Techo::g_tej_window = nullptr;
	if (auto* w = g_worker.exchange(nullptr, std::memory_order_acq_rel)) delete w;
}

// Hilo del juego: un envio de la cola 0.
inline void OnEnqueue(uint32_t queue_id, uint64_t serial, Worker::Job::Kind kind, bool reset, std::span<const uint32_t> commands) {
	if (queue_id != 0 || Mode() == 0) return;
	auto* w = g_worker.load(std::memory_order_acquire);
	if (w == nullptr) return;
	// (Ventana A: nada se copia; el serial no seguido rompe la cadena, que se reengancha en la B.)
	if (!Active()) return;
	w->Push(serial, kind, reset, commands);
}

// Hilo de render: la DCB de la cola 0 que traduce (una por vez, en orden; varias rebanadas si para en esperas).
struct GpuDcb {
	uint64_t serial   = 0;
	Slot*    slot     = nullptr;
	uint32_t next     = 0;
	uint64_t prev_tsc = 0;
	uint64_t draws    = 0;
	bool     on       = false;
};
inline GpuDcb g_gpu; // (solo el hilo de render)

struct GpuSlice {
	bool set = false;
	GpuSlice(bool graphics_q0, uint64_t serial, bool first_slice) {
		if (Mode() == 0 || !graphics_q0) return;
		if (first_slice) {
			g_gpu_serial.store(serial, std::memory_order_release);
			g_render_cpu.store(LocalPlatform::CurrentCpu(), std::memory_order_relaxed);
			g_gpu        = {};
			g_gpu.serial = serial;
			g_gpu.on     = Active();
			if (g_gpu.on) {
				Add(GDcbs);
				if (auto* w = g_worker.load(std::memory_order_acquire); w != nullptr) {
					auto& s = w->SlotOf(serial);
					if (s.serial.load(std::memory_order_acquire) == serial) {
						g_gpu.slot = &s;
						Add(GDcbSlot);
					}
				}
			}
		}
		if (g_gpu.serial != serial || !g_gpu.on) return;
		t_side = 1;
		set    = true;
	}
	~GpuSlice() {
		if (!set) return;
		t_side = 0;
		Add(GDraws, std::exchange(g_gpu.draws, 0));
	}
	GpuSlice(const GpuSlice&)            = delete;
	GpuSlice& operator=(const GpuSlice&) = delete;
};

// Hilo de render: el primer draw de cada paquete de la sombra (la misma posicion: clave = desplazamiento en la DCB
// raiz << 32 | en el bufer actual), comparado; lo publicado y no visto, tarde o saltado.
inline void GpuDraw(const CommandProcessor& cp, RenderContext& r, uint64_t key) {
	auto& c = g_gpu;
	++c.draws;
	Slot* const s = c.slot;
	if (s == nullptr) return;
	const uint64_t now = __rdtsc();
	const uint32_t n   = s->count.load(std::memory_order_acquire);
	while (c.next < n) {
		const Packet& p = s->pk[c.next];
		if (p.key > key) break;
		++c.next;
		if (p.key < key) { // (paso sin verlo: publicado despues del draw anterior = tarde; antes = el render no paso por ahi)
			Add(p.publish_tsc > c.prev_tsc ? GPkLate : GPkSkip);
			continue;
		}
		const uint64_t t0   = __rdtsc();
		const bool     same = cp.SameState(*p.cp);
		const bool     eb   = r.GetBufferCache().RegistrationEpoch() == p.ep_buf;
		const bool     ei   = r.GetTextureCache().ResolutionEpoch() == p.ep_img;
		const bool     em   = r.GetGpuResources().MappingEpoch() == p.ep_map;
		Add(GValTicks, __rdtsc() - t0);
		Add(GValN);
		const uint32_t sewn = p.draws > p.holes ? p.draws - p.holes : 0;
		if (!same) {
			Add(GPkRegs);
		} else {
			Add(GDrawsRegsOk, sewn);
			if (!eb) Add(GPkEpBuf);
			else if (!ei) Add(GPkEpImg);
			else if (!em) Add(GPkEpMap);
			else {
				Add(GPkOk);
				Add(GDrawsOk, sewn);
				Add(GHolesOk, p.holes);
			}
		}
		break;
	}
	c.prev_tsc = now;
}

inline void OnDraw(const CommandProcessor& cp, RenderContext& r, uint64_t key) {
	if (t_side == 1) GpuDraw(cp, r, key);
	else if (auto* w = g_worker.load(std::memory_order_relaxed); w != nullptr) w->WorkerDraw(cp, r, key);
}

// Hilo de render: un envio de la cola 0 acabado (estado para reenganchar la cadena; cola de la DCB tarde).
inline void GpuDone(uint64_t serial, const CommandProcessor& cp) {
	if (Mode() == 0) return;
	if (g_gpu.serial == serial && g_gpu.slot != nullptr) {
		if (!g_gpu.slot->done.load(std::memory_order_acquire)) Add(GTailLate);
		g_gpu.slot = nullptr;
	}
	if (auto* w = g_worker.load(std::memory_order_acquire); w != nullptr) w->GpuDone(serial, cp);
}

// Ventana de 5 s de [techo] (hilo del informe).
inline void Window(double busy_ms, double fps, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	const bool was   = g_ab_on.load(std::memory_order_relaxed);
	double     delta = 0, cnt[2] {};
	if (m == 2) {
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
	uint64_t v[NCount];
	for (uint32_t i = 0; i < NCount; ++i) v[i] = g_c[i].exchange(0, std::memory_order_relaxed);
	// (TSC -> ns, calibrado entre ventanas)
	static uint64_t last_tsc = 0;
	static auto     last_t   = std::chrono::steady_clock::now();
	const uint64_t  tsc      = __rdtsc();
	const auto      t        = std::chrono::steady_clock::now();
	const double    ns_per_tick =
        last_tsc != 0 && tsc > last_tsc ? std::chrono::duration<double, std::nano>(t - last_t).count() / static_cast<double>(tsc - last_tsc) : 0.25;
	last_tsc = tsc, last_t = t;
	const double f = frames;
	const double draws = static_cast<double>(v[GDraws]);
	const auto   pct   = [&](uint64_t x) { return draws > 0 ? 100.0 * static_cast<double>(x) / draws : 0.0; };
	const double on_time = static_cast<double>(v[GPkOk] + v[GPkRegs] + v[GPkEpBuf] + v[GPkEpImg] + v[GPkEpMap]);
	const double seen_pk = on_time + static_cast<double>(v[GPkLate]);
	const double val_ns  = v[GValN] != 0 ? static_cast<double>(v[GValTicks]) * ns_per_tick / static_cast<double>(v[GValN]) : 0;
	const double val_per_sewn =
	    v[GDrawsOk] != 0 ? static_cast<double>(v[GValTicks]) * ns_per_tick / static_cast<double>(v[GDrawsOk]) : 0;
	const double w_us = v[WDraws] != 0 ? static_cast<double>(v[WNs]) / 1e3 / static_cast<double>(v[WDraws]) : 0;
	const uint64_t fn = g_free_n.exchange(0, std::memory_order_relaxed), fs = g_free_sum.exchange(0, std::memory_order_relaxed);
	std::printf("[tej] ventana %s: render ocupado %.2f ms, %.1f fps | por fot: DCB %.1f (con sombra %.1f), draws %.0f |"
	            " COSIBLES %.1f %% (%.0f/fot; cota solo registros %.1f %%) | paquetes a tiempo %.1f %% (%.1f/fot validados, tarde"
	            " %.1f, saltados %.1f, DCB acabadas antes que la sombra %.1f) | fallo de paquete por fot: registros %.1f,"
	            " epoca buffers %.1f, imagenes %.1f, mapeos %.1f | validacion %.0f ns/paquete = %.0f ns/llamada cosida |"
	            " trabajador %.2f us/llamada (%.0f draws/fot, %.2f ms/fot), huecos draw %.1f/fot (otros %.1f), paradas %.1f,"
	            " fallos %.1f, +64*%d %.1f | sin traducir por fot: sin nucleo libre %.1f, render ya paso %.1f, sin estado %.1f"
	            " (publicados %.1f), otro tipo %.1f, cola llena %.1f; espera adelanto %.2f ms/fot; copia %.0f KiB/fot;"
	            " reafinidad %.1f/fot | CPU libre %.2f nucleos",
	            m == 2 ? (was ? "B (sombra)" : "A (apagada)") : "sombra", busy_ms, fps, v[GDcbs] / f, v[GDcbSlot] / f, draws / f,
	            pct(v[GDrawsOk]), v[GDrawsOk] / f, pct(v[GDrawsRegsOk]), seen_pk > 0 ? 100.0 * on_time / seen_pk : 0.0,
	            on_time / f, v[GPkLate] / f, v[GPkSkip] / f, v[GTailLate] / f, v[GPkRegs] / f, v[GPkEpBuf] / f,
	            v[GPkEpImg] / f, v[GPkEpMap] / f, val_ns, val_per_sewn, w_us, v[WDraws] / f, v[WNs] / 1e6 / f,
	            v[WHolesDraw] / f, v[WHolesOther] / f, v[WStops] / f, v[WFailed] / f, static_cast<int>(MaxPackets),
	            v[WOver] / f, v[WNoCpu] / f, v[WBehind] / f, v[WNoState] / f, v[GPublish] / f, v[WOther] / f,
	            v[WDropFull] / f, v[WAheadWaitNs] / 1e6 / f, v[WCopyBytes] / 1024.0 / f, v[WRepins] / f,
	            fn != 0 ? static_cast<double>(fs) / 100.0 / static_cast<double>(fn) : -1.0);
	if (m == 2)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot (objetivo 0 +-0.2)", cnt[0], cnt[1], delta);
	std::printf("\n");
	if (auto* w = g_worker.load(std::memory_order_acquire); w != nullptr) {
		std::vector<std::pair<const char*, uint64_t>> holes, stops;
		w->TakeReasons(holes, stops);
		const auto top = [&](const char* title, std::vector<std::pair<const char*, uint64_t>>& list) {
			std::ranges::sort(list, [](const auto& a, const auto& b) { return a.second > b.second; });
			std::printf(" %s:", title);
			for (size_t i = 0; i < list.size() && i < 8; ++i) std::printf(" [%s] %.1f", list[i].first, list[i].second / f);
		};
		if (!holes.empty() || !stops.empty()) {
			std::printf("[tej] por fot");
			top("huecos (draws) por motivo", holes);
			top("| paradas", stops);
			std::printf("\n");
		}
	}
	std::fflush(stdout);
	if (m == 2) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace Tej
