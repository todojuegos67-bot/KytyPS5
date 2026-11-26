// (juego3) KYTY_JUEGO3: the game thread's time off the CPU, by path, and the readback begun at the label (local/juego3.h).

#include "local/juego3.h"

#include "local/local-platform.h"
#include "local/parpadeo7.h" // (the prefetch's counters, shared with KYTY_PARPADEO7)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace Juego3 {

namespace {

constexpr int64_t WindowNs = 5'000'000'000ll;
constexpr uint64_t EbootBegin = 0x900000000ull, EbootEnd = 0x904000000ull; // (01.007.000: the crash logs' "eboot.bin+")

const char* const TagName[TagCount] = {"sin etiqueta", "asentado 0xa1f290", "accesor 0x909940/70+0xa1f020", "lectura de vuelta",
                                       "espera al hilo GPU", "fallo de pagina"};

std::atomic<uint64_t> g_tag_ns[TagCount] {};
std::atomic<uint64_t> g_tag_n[TagCount] {};

std::atomic<uint64_t> g_settle_n {0}, g_settle_rb {0}, g_settle_us {0}, g_settle_game_n {0}, g_settle_game_us {0};

// ---- frames (the game thread writes, the sampler reads) ----
struct Frame {
	float   period_ms = 0, build_ms = 0, cpu_ms = -1;
	uint8_t on = 0, ccd1 = 0;
};
std::mutex         g_frames_mutex;
std::vector<Frame> g_frames;
std::atomic<double> g_tsc_per_ns {0.0};
std::atomic<uint32_t> g_thread_changes {0};
std::atomic<bool>  g_sampler_started {false};

struct GameState {
	int64_t  enter = 0, last_exit = 0;
	uint64_t last_cycles = 0;
	uint32_t last_tid = 0;
};
GameState g_game; // (the game thread only)

// ---- learning ring and the fix ----
struct WriteSlot {
	std::atomic<uint64_t> shader {0}, begin {0}, end {0};
};
constexpr size_t                      RingSize = 16384;
std::array<WriteSlot, RingSize>       g_ring;
std::atomic<uint64_t>                 g_ring_next {0};
constexpr size_t                      MaxLearned = 32;
std::array<std::atomic<uint64_t>, MaxLearned> g_learned {};
std::atomic<uint32_t>                 g_learned_n {0};
std::atomic<int64_t>                  g_last_lookup {0};
std::atomic<uint64_t>                 g_lookups {0}, g_learn_hits {0};
std::mutex                            g_pending_mutex;
std::vector<std::pair<uint64_t, uint64_t>> g_pending;
std::atomic<uint64_t>                 g_noted {0}, g_dropped {0}, g_labels {0}, g_spans {0}, g_span_bytes {0};

bool Learned(uint64_t shader) noexcept {
	const uint32_t n = g_learned_n.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < n && i < MaxLearned; ++i)
		if (g_learned[i].load(std::memory_order_relaxed) == shader) return true;
	return false;
}

double Median(std::vector<double>& v) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}
double Pct(std::vector<double>& v, double p) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

} // namespace

int ModeSlow() noexcept {
	const char* v = std::getenv("KYTY_JUEGO3");
	int         m = 0;
	if (v != nullptr && v[0] >= '0' && v[0] <= '3' && v[1] == 0) m = v[0] - '0';
	int expected = -1;
	g_mode.compare_exchange_strong(expected, m, std::memory_order_relaxed);
	if (m == 2) g_fix_now.store(true, std::memory_order_relaxed);
	return g_mode.load(std::memory_order_relaxed);
}

uint32_t CallerTid() noexcept {
	thread_local uint32_t tid = 0;
	if (tid == 0) tid = LocalPlatform::ThreadId();
	return tid;
}

int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void ScopeEnd(uint32_t tag, int64_t t0) noexcept {
	const int64_t ns = NowNs() - t0;
	if (tag >= TagCount || ns <= 0) return;
	g_tag_ns[tag].fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
	g_tag_n[tag].fetch_add(1, std::memory_order_relaxed);
}

void NoteSettle(uint64_t us, bool read_back) noexcept {
	if (Mode() == 0) return;
	g_settle_n.fetch_add(1, std::memory_order_relaxed);
	g_settle_us.fetch_add(us, std::memory_order_relaxed);
	if (read_back) g_settle_rb.fetch_add(1, std::memory_order_relaxed);
	if (IsGameThread()) {
		g_settle_game_n.fetch_add(1, std::memory_order_relaxed);
		g_settle_game_us.fetch_add(us, std::memory_order_relaxed);
	}
}

// ---- the fix ----
void NoteWriteSlow(uint64_t shader, uint64_t vaddr, uint64_t size) noexcept {
	const uint64_t i = g_ring_next.fetch_add(1, std::memory_order_relaxed);
	auto&          s = g_ring[i % RingSize];
	s.shader.store(0, std::memory_order_relaxed);
	s.begin.store(vaddr, std::memory_order_relaxed);
	s.end.store(vaddr + size, std::memory_order_relaxed);
	s.shader.store(shader, std::memory_order_release);
	if (!FixNow() || !Learned(shader)) return;
	try {
		std::lock_guard lock(g_pending_mutex);
		if (g_pending.size() >= 512) {
			g_dropped.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		g_pending.emplace_back(vaddr, size);
		g_pending_n.store(static_cast<uint32_t>(g_pending.size()), std::memory_order_relaxed);
		g_noted.fetch_add(1, std::memory_order_relaxed);
	} catch (...) {
	}
}

void LearnReadback(uint64_t vaddr, uint64_t size) noexcept {
	const int m = Mode();
	if ((m != 1 && m != 2) || size == 0) return;
	// At most one lookup every 20 ms (the ring scan is ~16K loads): the culling's shaders show up within a frame or two.
	const int64_t now  = NowNs();
	int64_t       last = g_last_lookup.load(std::memory_order_relaxed);
	if (now - last < 20'000'000 || !g_last_lookup.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
	g_lookups.fetch_add(1, std::memory_order_relaxed);
	const uint64_t end   = vaddr + size;
	const uint64_t next  = g_ring_next.load(std::memory_order_acquire);
	const uint64_t first = next > RingSize ? next - RingSize : 0;
	for (uint64_t i = next; i > first; --i) {
		const auto&    s      = g_ring[(i - 1) % RingSize];
		const uint64_t shader = s.shader.load(std::memory_order_acquire);
		if (shader == 0) continue;
		if (s.begin.load(std::memory_order_relaxed) >= end || vaddr >= s.end.load(std::memory_order_relaxed)) continue;
		g_learn_hits.fetch_add(1, std::memory_order_relaxed);
		if (Learned(shader)) return;
		const uint32_t n = g_learned_n.load(std::memory_order_relaxed);
		if (n >= MaxLearned) return;
		g_learned[n].store(shader, std::memory_order_relaxed);
		g_learned_n.store(n + 1, std::memory_order_release);
		std::printf("[juego3] aprendido shader de computo 0x%" PRIx64 " (escribio 0x%" PRIx64 "+0x%" PRIx64
		            " que el juego leyo de vuelta); %u en total\n",
		            shader, vaddr, size, n + 1);
		std::fflush(stdout);
		return;
	}
}

bool TakePendingSlow(std::vector<std::pair<uint64_t, uint64_t>>& out) noexcept {
	out.clear();
	try {
		{
			std::lock_guard lock(g_pending_mutex);
			if (g_pending.empty()) return false;
			out.swap(g_pending);
			g_pending_n.store(0, std::memory_order_relaxed);
		}
		Parpadeo7::JoinRanges(out); // (sorted, joined: <4 KiB apart, spans <= 256 KiB)
		// At most 16 spans and 8 MiB a label (a guest readback slot holds a 512 KiB window; PrefetchGuestRead leaves 8 of its 32
		// slots free).
		uint64_t total = 0;
		size_t   n     = 0;
		for (; n < out.size() && n < 16; ++n) {
			if (total + out[n].second > (uint64_t {8} << 20u)) break;
			total += out[n].second;
		}
		out.resize(n);
		g_labels.fetch_add(1, std::memory_order_relaxed);
		g_spans.fetch_add(n, std::memory_order_relaxed);
		g_span_bytes.fetch_add(total, std::memory_order_relaxed);
		return n != 0;
	} catch (...) {
		out.clear();
		return false;
	}
}

// ---- the sampler and the lines ----
namespace {

struct Site {
	uint32_t tag = 0, kernel = 0;
	uint64_t guest = 0, exe1 = 0, exe2 = 0;
	uint64_t count = 0;
};
struct Sites {
	std::array<Site, 160> s {};
	size_t                n     = 0;
	uint64_t              total = 0, lost = 0;
	void Add(const Site& k) {
		++total;
		for (size_t i = 0; i < n; ++i)
			if (s[i].tag == k.tag && s[i].kernel == k.kernel && s[i].guest == k.guest && s[i].exe1 == k.exe1 && s[i].exe2 == k.exe2) {
				++s[i].count;
				return;
			}
		if (n == s.size()) {
			++lost;
			return;
		}
		s[n] = k;
		s[n].count = 1;
		++n;
	}
	void Clear() {
		n = 0, total = 0, lost = 0;
	}
	void Print(const char* title, double frames) const {
		std::array<size_t, 160> order {};
		for (size_t i = 0; i < n; ++i) order[i] = i;
		std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n), [&](size_t a, size_t b) { return s[a].count > s[b].count; });
		std::printf("[juego3s] %s: %llu muestras de pila del hilo del juego fuera de CPU (%.2f/frame; %llu sin hueco); sitio = etiqueta | "
		            "rip en ntdll (bloqueado en el nucleo) o no (listo, expulsado) | primer retorno al juego | primeros marcos del "
		            "emulador (exe+0x, ver el .map) -> %% de las muestras:",
		            title, static_cast<unsigned long long>(total), frames > 0 ? static_cast<double>(total) / frames : 0.0,
		            static_cast<unsigned long long>(lost));
		for (size_t j = 0; j < n && j < 10; ++j) {
			const auto& e = s[order[j]];
			std::printf(" | %s %s eboot+0x%" PRIx64 " exe+0x%" PRIx64 " exe+0x%" PRIx64 " %.1f%%", TagName[e.tag],
			            e.kernel != 0 ? "nucleo" : "LISTO", e.guest != 0 ? e.guest - EbootBegin : 0, e.exe1, e.exe2,
			            total != 0 ? 100.0 * static_cast<double>(e.count) / static_cast<double>(total) : 0.0);
		}
		std::printf("\n");
	}
};

struct WindowSummary {
	bool   valid = false, on = false;
	double frame_p50 = 0, build_p50 = 0, off_p50 = 0, cpu_p50 = 0;
};

struct Sampler {
	uint64_t handle = 0, gpu_handle = 0, gpu_c0 = 0;
	uint64_t exe_begin = 0, exe_end = 0, ntdll_begin = 0, ntdll_end = 0;
	// per window
	double   off_ns[TagCount] {}, on_ns[TagCount] {}, total_ns = 0;
	uint64_t stacks = 0, stack_fail = 0;
	Sites    win_sites, run_sites;
	std::vector<WindowSummary> history;
	uint64_t prefetch_last[8] {};
	uint64_t windows = 0;
};

void Calibrate() {
	const auto     t0 = std::chrono::steady_clock::now();
	const uint64_t c0 = __rdtsc();
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	const uint64_t c1 = __rdtsc();
	const double   ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
	if (ns > 0 && c1 > c0) g_tsc_per_ns.store(static_cast<double>(c1 - c0) / ns, std::memory_order_relaxed);
}

bool ThreadCycles(uint64_t handle, uint64_t* out) {
#if defined(_WIN32)
	ULONG64 c = 0;
	if (QueryThreadCycleTime(reinterpret_cast<HANDLE>(handle), &c) == 0) return false;
	*out = c;
	return true;
#else
	(void)handle, (void)out;
	return false;
#endif
}

[[maybe_unused]] void ModuleRange(const char* name, uint64_t* begin, uint64_t* end) {
#if defined(_WIN32)
	HMODULE module = name == nullptr ? GetModuleHandleA(nullptr) : GetModuleHandleA(name);
	MODULEINFO info {};
	if (module != nullptr && GetModuleInformation(GetCurrentProcess(), module, &info, sizeof(info)) != 0) {
		*begin = reinterpret_cast<uint64_t>(info.lpBaseOfDll);
		*end   = *begin + info.SizeOfImage;
	}
#else
	(void)name, (void)begin, (void)end;
#endif
}

void PrintWindow(Sampler& sm, int64_t seconds_ns, bool on, int mode) {
	std::vector<Frame> frames;
	{
		std::lock_guard lock(g_frames_mutex);
		frames.swap(g_frames);
	}
	std::vector<double> period, build, cpu, off;
	uint64_t            ccd1 = 0;
	period.reserve(frames.size()), build.reserve(frames.size()), cpu.reserve(frames.size()), off.reserve(frames.size());
	double max_period = 0;
	for (const auto& f: frames) {
		period.push_back(f.period_ms);
		build.push_back(f.build_ms);
		max_period = std::max(max_period, static_cast<double>(f.period_ms));
		if (f.cpu_ms >= 0) {
			cpu.push_back(f.cpu_ms);
			off.push_back(std::max(0.0, static_cast<double>(f.build_ms - f.cpu_ms)));
		}
		ccd1 += f.ccd1;
	}
	const double n = static_cast<double>(frames.size());
	const double p50 = Median(period), p90 = Pct(period, 0.9), p99 = Pct(period, 0.99);
	const double b50 = Median(build), c50 = Median(cpu), o50 = Median(off);
	WindowSummary w;
	w.on        = on;
	w.frame_p50 = p50, w.build_p50 = b50, w.off_p50 = o50, w.cpu_p50 = c50;
	// A loading or hitching window is not compared: fewer than 20 fps, a frame over 100 ms, or p99 over 2.5x p50.
	w.valid = frames.size() >= 100 && max_period < 100.0 && p99 < 2.5 * p50;
	sm.history.push_back(w);
	const auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	double     tag_ms[TagCount] {}, tag_n[TagCount] {};
	for (uint32_t t = 0; t < TagCount; ++t) {
		tag_ms[t] = static_cast<double>(take(g_tag_ns[t])) / 1e6;
		tag_n[t]  = static_cast<double>(take(g_tag_n[t]));
	}
	const double settle_n = static_cast<double>(take(g_settle_n)), settle_rb = static_cast<double>(take(g_settle_rb)),
	             settle_ms = static_cast<double>(take(g_settle_us)) / 1e3, settle_gn = static_cast<double>(take(g_settle_game_n)),
	             settle_gms = static_cast<double>(take(g_settle_game_us)) / 1e3;
	const double noted = static_cast<double>(take(g_noted)), dropped = static_cast<double>(take(g_dropped)),
	             labels = static_cast<double>(take(g_labels)), spans = static_cast<double>(take(g_spans)),
	             span_kib = static_cast<double>(take(g_span_bytes)) / 1024.0, lookups = static_cast<double>(take(g_lookups)),
	             hits = static_cast<double>(take(g_learn_hits));
	// The parpadeo7 counters the prefetch and the settle share (read, not reset: KYTY_PARPADEO7's line resets its own).
	auto&          c = Parpadeo7::g_c;
	const uint64_t now_c[8] = {c.begun.load(), c.inflight.load(), c.no_slot.load(), c.not_gpu.load(), c.refused.load(),
	                           c.settle_in.load(), c.settle_late.load(), c.settle_miss.load()};
	uint64_t       d[8] {};
	for (int i = 0; i < 8; ++i) {
		d[i]               = now_c[i] >= sm.prefetch_last[i] ? now_c[i] - sm.prefetch_last[i] : now_c[i];
		sm.prefetch_last[i] = now_c[i];
	}
	const double per = n > 0 ? 1.0 / n : 0.0;
	// The GPU thread's CPU in the window (its busy time: f0 measured busy ~= CPU there).
	double   gpu_ms = -1;
	uint64_t gc     = 0;
	if (sm.gpu_handle == 0 && g_gpu_tid.load(std::memory_order_relaxed) != 0) {
		sm.gpu_handle = LocalPlatform::OpenThreadForSampling(g_gpu_tid.load(std::memory_order_relaxed));
		if (sm.gpu_handle != 0 && !ThreadCycles(sm.gpu_handle, &sm.gpu_c0)) sm.gpu_c0 = 0;
	} else if (sm.gpu_handle != 0 && ThreadCycles(sm.gpu_handle, &gc)) {
		const double tpn = g_tsc_per_ns.load(std::memory_order_relaxed);
		if (sm.gpu_c0 != 0 && gc > sm.gpu_c0 && tpn > 0 && n > 0) gpu_ms = static_cast<double>(gc - sm.gpu_c0) / tpn / 1e6 / n;
		sm.gpu_c0 = gc;
	}
	double       off_total = 0;
	for (double v: sm.off_ns) off_total += v;
	std::printf("[juego3] %.1f s, VENTANA %s (KYTY_JUEGO3=%d), %.0f frames%s | frame p50 %.2f p90 %.2f p99 %.2f ms | hilo del juego "
	            "(suspend->suspend): construye p50 %.2f, CPU p50 %.2f (mismo hilo, QueryThreadCycleTime), FUERA DE CPU p50 %.2f ms; acabados en CPUs 16-31 "
	            "(CCD1) %.0f%% de los frames | hilo GPU del emulador CPU %.2f ms/frame | FUERA DE CPU por muestreo (1 ms) %.2f ms/frame =",
	            static_cast<double>(seconds_ns) / 1e9, on ? "ON" : "OFF", mode, n, w.valid ? "" : " (NO COMPARABLE: carga o tirones)", p50,
	            p90, p99, b50, c50, o50, n > 0 ? 100.0 * static_cast<double>(ccd1) / n : 0.0, gpu_ms, off_total / 1e6 * per);
	for (uint32_t t = 0; t < TagCount; ++t) std::printf("%s %s %.2f", t == 0 ? "" : " +", TagName[t], sm.off_ns[t] / 1e6 * per);
	std::printf(" | EN CPU dentro de cada una ms/frame:");
	for (uint32_t t = 1; t < TagCount; ++t) std::printf(" %s %.2f", TagName[t], sm.on_ns[t] / 1e6 * per);
	std::printf(" | reloj de cada via (incl.) ms/frame [veces/frame]:");
	for (uint32_t t = 1; t < TagCount; ++t) std::printf(" %s %.2f [%.1f]", TagName[t], tag_ms[t] * per, tag_n[t] * per);
	std::printf(" | asentados 0xa1f290 (todos los hilos)/frame: %.1f, con lectura de vuelta %.2f, %.2f ms (del hilo del juego %.1f, %.2f ms)",
	            settle_n * per, settle_rb * per, settle_ms * per, settle_gn * per, settle_gms * per);
	std::printf(" | ARREGLO: shaders aprendidos %u (busquedas %.0f, con escritor %.0f); escrituras anotadas %.1f/frame (perdidas %.0f), "
	            "etiquetas con precarga %.1f/frame, tramos %.1f/frame, %.0f KiB/frame; precargas: empezadas %.1f, ya en vuelo %.1f, sin hueco "
	            "%.1f, nada de GPU %.1f, rechazadas %.1f /frame; asentados que encontraron: ya dentro %.1f, en vuelo %.1f, SIN PRECARGA %.1f "
	            "/frame | coste: muestras de pila %llu (fallidas %llu); cambios de hilo del juego %u\n",
	            g_learned_n.load(), lookups, hits, noted * per, dropped, labels * per, spans * per, span_kib * per,
	            static_cast<double>(d[0]) * per, static_cast<double>(d[1]) * per, static_cast<double>(d[2]) * per,
	            static_cast<double>(d[3]) * per, static_cast<double>(d[4]) * per, static_cast<double>(d[5]) * per,
	            static_cast<double>(d[6]) * per, static_cast<double>(d[7]) * per, static_cast<unsigned long long>(sm.stacks),
	            static_cast<unsigned long long>(sm.stack_fail), g_thread_changes.load());
	sm.win_sites.Print(on ? "ventana ON" : "ventana OFF", n);
	std::fflush(stdout);
	for (auto& v: sm.off_ns) v = 0;
	for (auto& v: sm.on_ns) v = 0;
	sm.total_ns = 0, sm.stacks = 0, sm.stack_fail = 0;
	sm.win_sites.Clear();
	++sm.windows;
	// Every 60 s: ON - OFF over OFF/ON/OFF (and ON/OFF/ON) triplets of consecutive comparable windows whose two outer
	// windows differ by less than 8% (the same scene); mode 1 only.
	if (mode != 1 || sm.windows % 12 != 0) return;
	std::vector<double> df, db, doff, dcpu;
	const auto&         h = sm.history;
	for (size_t i = 1; i + 1 < h.size(); ++i) {
		const auto &a = h[i - 1], &m = h[i], &b = h[i + 1];
		if (!a.valid || !m.valid || !b.valid || a.on != b.on || a.on == m.on) continue;
		if (std::fabs(a.frame_p50 - b.frame_p50) > 0.08 * std::min(a.frame_p50, b.frame_p50)) continue;
		const double sign = m.on ? 1.0 : -1.0; // ON - OFF
		df.push_back(sign * (m.frame_p50 - 0.5 * (a.frame_p50 + b.frame_p50)));
		db.push_back(sign * (m.build_p50 - 0.5 * (a.build_p50 + b.build_p50)));
		doff.push_back(sign * (m.off_p50 - 0.5 * (a.off_p50 + b.off_p50)));
		dcpu.push_back(sign * (m.cpu_p50 - 0.5 * (a.cpu_p50 + b.cpu_p50)));
	}
	size_t valid = 0;
	for (const auto& x: h) valid += x.valid ? 1 : 0;
	const size_t k = df.size();
	std::printf("[juego3] RESUMEN (%zu ventanas, %zu comparables; %zu tripletes OFF/ON/OFF u ON/OFF/ON de la misma escena): ON - OFF "
	            "mediana: frame %+.2f ms (min %+.2f max %+.2f), construye %+.2f, FUERA DE CPU %+.2f, CPU %+.2f ms/frame%s\n",
	            h.size(), valid, k, Median(df), k ? *std::min_element(df.begin(), df.end()) : 0.0,
	            k ? *std::max_element(df.begin(), df.end()) : 0.0, Median(db), Median(doff), Median(dcpu),
	            k < 4 ? " (POCOS TRIPLETES: no concluir)" : "");
	sm.run_sites.Print("RESUMEN de la partida", 0.0);
	std::fflush(stdout);
}

void SamplerRun(uint32_t tid) {
	LocalPlatform::SetThreadName("Kyty.Juego3");
	Sampler sm;
	Calibrate();
#if defined(_WIN32)
	sm.handle = LocalPlatform::OpenThreadForSampling(tid);
	LocalPlatform::PrepareSampling();
	ModuleRange(nullptr, &sm.exe_begin, &sm.exe_end);
	ModuleRange("ntdll.dll", &sm.ntdll_begin, &sm.ntdll_end);
	{
		GROUP_AFFINITY affinity {};
		const bool     got = sm.handle != 0 && GetThreadGroupAffinity(reinterpret_cast<HANDLE>(sm.handle), &affinity) != 0;
		const char*    rc  = std::getenv("KYTY_RENDER_CPUS");
		std::printf("[juego3] hilo del juego %u: prioridad %d, afinidad 0x%llx (grupo %u)%s; KYTY_RENDER_CPUS=%s; TSC %.3f/ns\n", tid,
		            sm.handle != 0 ? GetThreadPriority(reinterpret_cast<HANDLE>(sm.handle)) : 0,
		            got ? static_cast<unsigned long long>(affinity.Mask) : 0ull, got ? affinity.Group : 0u, got ? "" : " (sin leer)",
		            rc != nullptr ? rc : "(sin)", g_tsc_per_ns.load());
		std::fflush(stdout);
	}
#else
	(void)tid;
#endif
	const int mode     = Mode();
	int64_t   start    = NowNs();
	int64_t   last     = start;
	int64_t   last_stk = 0;
	uint64_t  last_c   = 0;
	bool      have_c   = sm.handle != 0 && ThreadCycles(sm.handle, &last_c);
	uint64_t  index    = 0;
	bool      on       = mode == 2;
	g_fix_now.store(on, std::memory_order_relaxed);
	for (;;) {
#if defined(_WIN32)
		Sleep(1);
#else
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
		const int64_t now = NowNs();
		uint64_t      c   = 0;
		const double  tpn = g_tsc_per_ns.load(std::memory_order_relaxed);
		if (have_c && ThreadCycles(sm.handle, &c) && tpn > 0) {
			const double dt  = static_cast<double>(now - last);
			double       run = c >= last_c ? static_cast<double>(c - last_c) / tpn : 0.0;
			run              = std::min(run, dt);
			const uint32_t tag = std::min<uint32_t>(g_cur_tag.load(std::memory_order_relaxed), TagCount - 1);
			sm.off_ns[tag] += dt - run;
			sm.on_ns[tag] += run;
			sm.total_ns += dt;
			last_c = c;
			// Mostly off the CPU in the interval: its stack, at most every 5 ms.
			if ((dt - run) * 2.0 > dt && now - last_stk >= 5'000'000) {
				last_stk = now;
				uint64_t out[32] {};
				if (LocalPlatform::SampleThread(sm.handle, 0, 0, out, 32)) {
					++sm.stacks;
					Site k;
					k.tag    = std::min<uint32_t>(g_cur_tag.load(std::memory_order_relaxed), TagCount - 1);
					k.kernel = out[0] >= sm.ntdll_begin && out[0] < sm.ntdll_end ? 1 : 0;
					for (uint64_t a: out) {
						if (a >= EbootBegin && a < EbootEnd) {
							k.guest = a;
							break;
						}
					}
					for (uint64_t a: out) {
						if (a < sm.exe_begin || a >= sm.exe_end) continue;
						if (k.exe1 == 0) k.exe1 = a - sm.exe_begin;
						else if (k.exe2 == 0 && a - sm.exe_begin != k.exe1) {
							k.exe2 = a - sm.exe_begin;
							break;
						}
					}
					sm.win_sites.Add(k);
					sm.run_sites.Add(k);
				} else {
					++sm.stack_fail;
				}
			}
		} else if (sm.handle != 0) {
			have_c = ThreadCycles(sm.handle, &last_c);
		}
		last = now;
		if (now - start >= WindowNs) {
			PrintWindow(sm, now - start, on, mode);
			start = now;
			++index;
			if (mode == 1) on = (index & 1u) != 0; // OFF, ON, OFF, ...
			g_fix_now.store(on, std::memory_order_relaxed);
		}
	}
}

} // namespace

void GameSuspend(bool entering) noexcept {
	const int mode = Mode();
	if (mode == 0) return;
	const uint32_t tid = CallerTid();
	const int64_t  now = NowNs();
	auto&          g   = g_game;
	if (g_game_tid.load(std::memory_order_relaxed) != tid) {
		if (g_game_tid.exchange(tid, std::memory_order_relaxed) != 0) g_thread_changes.fetch_add(1, std::memory_order_relaxed);
	}
	if (entering) {
		g.enter = now;
		return;
	}
	if (!g_sampler_started.exchange(true, std::memory_order_relaxed)) {
		std::printf("[juego3] KYTY_JUEGO3=%d: %s. Lineas [juego3]/[juego3s] cada 5 s y RESUMEN cada 60 s.\n", mode,
		            mode == 1   ? "medida + arreglo (lectura de vuelta empezada en la etiqueta) en ventanas alternas de 5 s OFF/ON"
		            : mode == 2 ? "medida + arreglo siempre"
		                        : "solo medida");
		std::fflush(stdout);
		try {
			std::thread(SamplerRun, tid).detach();
		} catch (...) {
		}
	}
	const uint64_t cycles = LocalPlatform::CurrentThreadCycles();
	const double   tpn    = g_tsc_per_ns.load(std::memory_order_relaxed);
	if (g.last_exit != 0 && g.last_tid == tid && g.enter > g.last_exit) {
		Frame f;
		f.period_ms = static_cast<float>(static_cast<double>(now - g.last_exit) / 1e6);
		f.build_ms  = static_cast<float>(static_cast<double>(g.enter - g.last_exit) / 1e6);
		f.cpu_ms    = tpn > 0 && cycles > g.last_cycles && g.last_cycles != 0 ? static_cast<float>(static_cast<double>(cycles - g.last_cycles) / tpn / 1e6)
		                                                                     : -1.0f;
		f.on        = FixNow() ? 1 : 0;
#if defined(_WIN32)
		f.ccd1 = GetCurrentProcessorNumber() >= 16 ? 1 : 0; // (9950X3D: CPUs 0-15 the first CCD)
#endif
		try {
			std::lock_guard lock(g_frames_mutex);
			if (g_frames.size() < 16384) g_frames.push_back(f);
		} catch (...) {
		}
	}
	g.last_exit   = now;
	g.last_cycles = cycles;
	g.last_tid    = tid;
}

} // namespace Juego3
