#pragma once
// (par2, fase 1 del plan de traduccion paralela, tools/plan-paralelo.md §7) KYTY_PAR2 (apagado por defecto).
// Da a la traduccion de la cola grafica (DCB, cola 0) una estructura POR HILO -"carril"- y un envio EN ORDEN por
// ticket, todavia con UN SOLO hilo de traduccion. Fase 3 traducira DCB en carriles distintos sobre esto.
//
// Ghidra (eboot): FUN_00a18840 = job "SubmitJob" que encola RenderThread::KickoffSubmitJob (FUN_00a08ae0): espera las
// dependencias (FUN_008351b0 por cada una), envia cada DCB de la lista (param+0x82..0x84) con FUN_00911c90
// (sce::Agc::submitGraphics), despues cada ACB (param+0x100..0x102) con FUN_00915540, y si el modo es 1 el ultimo DCB
// con FUN_00911d60 (submit + flip). Cada DCB llega aqui como una Submission de la cola 0 con su serial: el orden guest
// es el serial, y es el ticket del carril (el envio en orden de fase 3 compara contra el).
//
// Un carril agrupa lo que un hilo de traduccion necesita para grabar sin compartir:
//   - executors: los RenderExecutor de draw/dispatch (cache de tablas/XPR y SUS descriptor pools de NativeXpr). Nulo =
//     los de RenderContext (DefaultRenderExecutor), sin RenderContext::SetThreadExecutors.
//   - recorder:  CommandScheduler::Recorder (VkCommandPool + cmd buffers propios, enviados con SubmitRecorded en orden).
//     Nulo = el buffer abierto del CommandScheduler (lo de siempre).
//   - upload:    sub-anillo de subidas/push. Nulo = el StreamBuffer comun.
//   - prep:      cola de preparacion de recursos (fase 2): subidas, registro en caches, transiciones de imagen, LRU.
//     Un carril "in_order" (el carril 0, el del hilo que envia) la aplica al momento: Prepare(f) == f().
//     Un carril especulativo (fase 3) la guarda y Commit la aplica, en orden de ticket, antes de su envio.
// Fase 1: solo existe el carril 0 y todo es identidad (los mismos objetos de siempre): la imagen no cambia por
// construccion; lo nuevo es el ciclo Begin(ticket) / Commit(ticket) por DCB con su comprobacion de orden y la
// medida de su coste, que es lo que fase 3 pagara por DCB.
//   1 encendido.  2 A/B cada ventana de 5 s de [techo] (necesita KYTY_TECHO=1): B = encendido; delta interpolado.
//   3 (fase 2, prueba) el carril 0 se comporta como un carril FUERA DE ORDEN: lo diferible se guarda y se aplica en
//     el Commit (al final de cada trozo, o antes si alguien va a leer ese estado: Drain). Mide el coste del commit
//     diferido; la imagen no puede cambiar (solo se difiere lo que no tiene efecto GPU dentro de la DCB).
//   4 A/B de 3 (B = carril diferido) con KYTY_TECHO=1.
// Fase 2: que se difiere y que no (ver Kind):
//   - Diferible (efecto solo en estado CPU que nadie lee dentro del trozo salvo el recolector/borrado, que hacen
//     Drain antes): LRU de BufferCache (TouchBuffer: m_lru_cache + bda_use_ms + GestorMem::Tocar).
//   - NO diferible en el carril 0 (contado como "inmediata"): subidas de BufferCache (UploadDirtyRanges graba la copia
//     en el buffer abierto ANTES del draw que la lee y marca la pagina limpia; diferirla al Commit la pondria detras
//     del draw en el mismo VkCommandBuffer, y el leer los bytes del juego despues de una escritura CP de la misma DCB
//     -DMA_DATA/WRITE_DATA- dio el "mundo negro" del 10/06), registro de buffers nuevos (devuelve el VkBuffer que el
//     draw ya usa). Para diferirlas hace falta que la DCB grabe en un Recorder propio (prologo = buffer abierto,
//     enviado antes): fase 3.
// Apagado: una carga de un static por trozo de DCB.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "local/techo.h"

namespace Par2 {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_PAR2");
		int         m = v != nullptr ? std::atoi(v) : 0;
		if (m < 0 || m > 4) m = 0;
		if ((m == 2 || m == 4) && !Techo::On()) {
			std::printf("[par2] KYTY_PAR2=%d sin KYTY_TECHO=1: se usa %d (siempre encendido)\n", m, m - 1);
			m = m - 1;
		}
		if (m != 0)
			std::printf("[par2] KYTY_PAR2=%d (%s): carril 0 por hilo + envio en orden por ticket, 1 hilo (%s)\n", m,
			            (m == 1 || m == 3) ? "encendido" : "A/B cada 5 s, B = encendido",
			            m >= 3 ? "fase 2: carril 0 DIFERIDO, prep aplicada en Commit" : "fase 1");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}

inline std::atomic<bool> g_ab_on {false}; // (=2) esta ventana usa el carril (B)

[[nodiscard]] inline bool Active() {
	const int m = Mode();
	return m == 1 || m == 3 || ((m == 2 || m == 4) && g_ab_on.load(std::memory_order_relaxed));
}
// (fase 2) El carril 0 difiere su preparacion (modo 3/4).
[[nodiscard]] inline bool Deferred() { return Mode() >= 3; }

[[nodiscard]] inline uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Por que un trozo de DCB no paso por el carril.
enum Why : uint32_t { WOff, WSpec, WPar1, WRecorder, WCount };
inline constexpr const char* WhyNames[WCount] = {"ventana A", "especulacion", "par1", "grabador ajeno"};

struct Counters {
	std::atomic<uint64_t> dcbs {0};      // DCB abiertas (primer trozo)
	std::atomic<uint64_t> slices {0};    // trozos (una DCB parada en un WAIT_REG_MEM vuelve)
	std::atomic<uint64_t> commits {0};   // DCB enviadas en orden
	std::atomic<uint64_t> disorder {0};  // ticket fuera de orden (fase 1: debe ser 0)
	std::atomic<uint64_t> reopened {0};  // trozo con otro ticket abierto sin enviar (debe ser 0)
	std::atomic<uint64_t> prepared {0};  // preparaciones pasadas por Prepare (fase 2)
	std::atomic<uint64_t> deferred {0};  // ... guardadas para el Commit (carril no en orden; fase 1: 0)
	std::atomic<uint64_t> immediate {0}; // (fase 2) preparaciones de estado compartido NO diferibles hechas en linea
	std::atomic<uint64_t> drains {0};    // (fase 2) Drain antes de un lector (recolector, borrado) o por cola llena
	std::atomic<uint64_t> commit_ns {0}; // (fase 2) aplicar lo guardado (solo con KYTY_TECHO=1)
	std::atomic<uint64_t> scope_ns {0};  // coste propio de Begin/Commit (solo con KYTY_TECHO=1)
	std::atomic<uint64_t> why[WCount] {};
};
inline Counters g_c;

inline void Note(std::atomic<uint64_t>& c, uint64_t n = 1) { c.fetch_add(n, std::memory_order_relaxed); }

// Un carril de traduccion. Los punteros nulos son "los de siempre" (ver arriba). Tipos opacos aqui: este header no
// arrastra el renderer; quien crea un carril (fase 3) rellena los punteros con sus objetos.
struct Lane {
	int   id       = 0;
	bool  in_order = true;     // el hilo que envia: Prepare corre al momento
	void* executors = nullptr; // const RenderContext::Executors*
	void* recorder  = nullptr; // CommandScheduler::Recorder*
	void* upload    = nullptr; // sub-anillo de subidas
	// Ticket de la DCB que graba (serial de la cola 0) y el ultimo enviado.
	uint64_t open_ticket = 0;
	uint64_t committed   = 0;
	// (fase 2) Preparaciones guardadas: se aplican en Commit en orden. Argumentos por valor (con dueno: nada de
	// punteros a la pila del llamador): `obj` (la cache) y `arg` (un objeto con almacenamiento estable).
	struct Prep {
		void (*run)(void*, void*) = nullptr;
		void* obj                 = nullptr;
		void* arg                 = nullptr;
	};
	static constexpr uint32_t PrepMax = 4096;
	Prep     prep[PrepMax] {};
	uint32_t prep_count = 0;
};

inline Lane g_lane0;                         // el del hilo de render (hilo GPU)
inline thread_local Lane* t_lane = nullptr; // el carril del hilo actual (nulo: fuera de una DCB del carril)

// (fase 2) Aplica, en orden, lo guardado del carril (Commit y Drain).
inline void ApplyPrep(Lane& lane) {
	if (lane.prep_count == 0) return;
	const uint64_t t0 = Techo::On() ? NowNs() : 0;
	for (uint32_t i = 0; i < lane.prep_count; ++i) lane.prep[i].run(lane.prep[i].obj, lane.prep[i].arg);
	lane.prep_count = 0;
	if (t0 != 0) Note(g_c.commit_ns, NowNs() - t0);
}

// (fase 2) Una preparacion DIFERIBLE de la DCB en curso: al momento en un carril en orden (o fuera de carril), y si
// no, guardada para su Commit. Cola llena: aplica lo guardado (en orden) y guarda esta.
inline void Prepare(void (*run)(void*, void*), void* obj, void* arg) {
	Lane* lane = t_lane;
	if (lane == nullptr || lane->in_order) {
		if (lane != nullptr) Note(g_c.prepared);
		run(obj, arg);
		return;
	}
	if (lane->prep_count >= Lane::PrepMax) {
		Note(g_c.drains);
		ApplyPrep(*lane);
	}
	lane->prep[lane->prep_count++] = {run, obj, arg};
	Note(g_c.deferred);
}

// (fase 2) Un lector del estado diferido (recolector, borrado de un buffer) va a mirarlo: aplica antes lo guardado del
// carril de este hilo. Fuera de carril o sin nada guardado: una carga thread_local.
inline void Drain() {
	Lane* lane = t_lane;
	if (lane == nullptr || lane->prep_count == 0) return;
	Note(g_c.drains);
	ApplyPrep(*lane);
}

// (fase 2) Un lector/recolector del estado diferido entero (GC de BufferCache): drena y, mientras dura, lo que haga
// (p. ej. sus propios toques de LRU en mitad de su recorrido) va al momento, como siempre.
class Immediate {
public:
	Immediate() noexcept {
		Drain();
		m_prev = std::exchange(t_lane, nullptr);
	}
	~Immediate() { t_lane = m_prev; }
	Immediate(const Immediate&)            = delete;
	Immediate& operator=(const Immediate&) = delete;

private:
	Lane* m_prev = nullptr;
};

// (fase 2) Una preparacion de estado compartido que el carril 0 no puede diferir (ver arriba): solo se cuenta, para
// saber cuanto tendra que resolver fase 3 (Recorder propio + prologo).
inline void NoteImmediate() {
	if (t_lane != nullptr) Note(g_c.immediate);
}

// Envio en orden: aplica lo preparado del carril y marca el ticket enviado. El llamador envia despues sus buffers
// (fase 3: SubmitRecorded del recorder del carril; fase 1: nada, el buffer abierto sigue su curso de siempre).
inline void Commit(Lane& lane, uint64_t ticket) {
	ApplyPrep(lane);
	// (El serial de la cola 0 cuenta tambien preparaciones de flip y fronteras de fotograma: con huecos; el orden es
	// estrictamente creciente.)
	if (ticket <= lane.committed) Note(g_c.disorder);
	lane.committed   = ticket;
	lane.open_ticket = 0;
	Note(g_c.commits);
}

// Un trozo de una DCB de la cola 0 en el hilo de render. `eligible`: cola 0 grafica. `spec`/`par1`/`recorder`: los
// modos que ya graban en otra estructura (se queda fuera, contado). Done(complete) antes de salir.
class DcbScope {
public:
	DcbScope(bool eligible, bool first_slice, uint64_t serial, bool spec, bool par1, bool recorder) {
		if (!eligible || Mode() == 0) return;
		const Why why = !Active() ? WOff : spec ? WSpec : par1 ? WPar1 : recorder ? WRecorder : WCount;
		if (why != WCount) {
			if (first_slice) Note(g_c.why[why]);
			return;
		}
		const uint64_t t0 = Techo::On() ? NowNs() : 0;
		m_lane            = &g_lane0;
		m_lane->in_order  = !Deferred();
		if (first_slice) {
			if (m_lane->open_ticket != 0) Note(g_c.reopened);
			m_lane->open_ticket = serial;
			Note(g_c.dcbs);
		} else if (m_lane->open_ticket != serial) {
			Note(g_c.reopened);
			m_lane->open_ticket = serial;
		}
		m_ticket = serial;
		m_prev   = std::exchange(t_lane, m_lane);
		Note(g_c.slices);
		if (t0 != 0) Note(g_c.scope_ns, NowNs() - t0);
	}
	void Done(bool complete) noexcept { m_complete = complete; }
	~DcbScope() {
		if (m_lane == nullptr) return;
		const uint64_t t0 = Techo::On() ? NowNs() : 0;
		t_lane            = m_prev;
		// Un trozo que vuelve sin acabar (WAIT_REG_MEM) aplica tambien lo suyo: entre trozos corren otras colas y el
		// recolector, que no deben ver estado a medias.
		if (m_complete) Commit(*m_lane, m_ticket);
		else ApplyPrep(*m_lane);
		if (t0 != 0) Note(g_c.scope_ns, NowNs() - t0);
	}
	DcbScope(const DcbScope&)            = delete;
	DcbScope& operator=(const DcbScope&) = delete;

private:
	Lane*    m_lane     = nullptr;
	Lane*    m_prev     = nullptr;
	uint64_t m_ticket   = 0;
	bool     m_complete = false;
};

// Ventana de 5 s de [techo] (hilo del informe).
inline void Window(double busy_ms, double frames, bool keep) {
	const int m = Mode();
	if (m == 0 || frames <= 0) return;
	auto take = [](std::atomic<uint64_t>& c) { return static_cast<double>(c.exchange(0, std::memory_order_relaxed)); };
	const double dcbs = take(g_c.dcbs), slices = take(g_c.slices), commits = take(g_c.commits), disorder = take(g_c.disorder),
	             reopened = take(g_c.reopened), prepared = take(g_c.prepared), deferred = take(g_c.deferred),
	             scope_ns = take(g_c.scope_ns), immediate = take(g_c.immediate), drains = take(g_c.drains),
	             commit_ns = take(g_c.commit_ns);
	double why[WCount];
	for (uint32_t i = 0; i < WCount; ++i) why[i] = take(g_c.why[i]);
	const bool was   = g_ab_on.load(std::memory_order_relaxed);
	double     delta = 0, cnt[2] {}, raw[2] {}, raw_n[2] {};
	if (m == 2 || m == 4) {
		struct W { bool b; double busy; };
		static W      win[4096];
		static size_t k = 0;
		if (keep && k < 4096) win[k++] = {was, busy_ms};
		double sum[2] {};
		for (size_t i = 1; i + 1 < k; ++i)
			if (win[i - 1].b != win[i].b && win[i + 1].b != win[i].b) {
				const double d = win[i].busy - 0.5 * (win[i - 1].busy + win[i + 1].busy);
				sum[win[i].b ? 1 : 0] += d, cnt[win[i].b ? 1 : 0] += 1;
			}
		const double db = cnt[1] > 0 ? sum[1] / cnt[1] : 0, da = cnt[0] > 0 ? sum[0] / cnt[0] : 0;
		delta = (cnt[0] > 0 && cnt[1] > 0) ? 0.5 * (db - da) : cnt[1] > 0 ? db : -da;
		// Medias simples de todas las ventanas A y B (sin interpolar): con pocas ventanas el delta interpolado tiene ruido
		// de +-1 ms (T70: +0.90 con B 17.05 < A 17.18); si las dos medidas no coinciden en signo, es ruido.
		for (size_t i = 0; i < k; ++i) raw[win[i].b ? 1 : 0] += win[i].busy, raw_n[win[i].b ? 1 : 0] += 1;
	}
	std::printf("[par2] ventana %s: render ocupado %.2f ms | DCB en carril %.1f/fot (trozos %.1f, enviadas en orden %.1f, "
	            "FUERA DE ORDEN %.0f, reabiertas %.0f) prep %.1f/fot guardadas %.0f | coste propio %.4f ms/fot | fuera (",
	            (m == 2 || m == 4) ? (was ? "B (carril)" : "A (normal)") : "carril", busy_ms, dcbs / frames, slices / frames,
	            commits / frames, disorder, reopened, prepared / frames, deferred, scope_ns / 1e6 / frames);
	for (uint32_t i = 0; i < WCount; ++i) std::printf("%s%s %.1f", i ? ", " : "", WhyNames[i], why[i] / frames);
	std::printf(")");
	std::printf(" | fase2: inmediatas (no diferibles) %.1f/fot, drenajes %.1f/fot, aplicar guardadas %.4f ms/fot (objetivo < 1)",
	            immediate / frames, drains / frames, commit_ns / 1e6 / frames);
	if (m == 2 || m == 4)
		std::printf(" | ventanas A %.0f B %.0f | delta interpolado B-A %+.3f ms/fot | medias A %.2f (%.0f) B %.2f (%.0f) B-A %+.3f "
		            "(objetivo 0 +-0.2; fiable si las dos coinciden)",
		            cnt[0], cnt[1], delta, raw_n[0] > 0 ? raw[0] / raw_n[0] : 0.0, raw_n[0], raw_n[1] > 0 ? raw[1] / raw_n[1] : 0.0,
		            raw_n[1], raw_n[0] > 0 && raw_n[1] > 0 ? raw[1] / raw_n[1] - raw[0] / raw_n[0] : 0.0);
	std::printf("\n");
	std::fflush(stdout);
	if (m == 2 || m == 4) g_ab_on.store(!was, std::memory_order_relaxed);
}

} // namespace Par2
