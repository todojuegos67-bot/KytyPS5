#ifndef EMULATOR_SRC_LOCAL_F5P1_TECHO_H_
#define EMULATOR_SRC_LOCAL_F5P1_TECHO_H_
// (f5p1 techo) KYTY_F5P1_TECHO=1 (off by default; Demon's Souls 01.007.000 only). Measurement only, test folder only:
// the real ceiling of the F5 pilot pass (the XPR depth pass: cascades eboot+0xa13320 pass 0 and DepthPrePass
// eboot+0xa13110 pass 1, both through 0xab57d0 -> 0xa1f290 -> the mesh draw loop 0xad8bc0).
//
// The game's side (loader/demonsSoulsCopy.cpp, render1's hook of 0xad8bc0, installed also for this probe): after the
// game's function ran whole, for pass 0 / 1 (pass descriptor +0x20), the range [begin, end) of the command buffer it
// wrote (cursor before and after, one block) is registered with a signature: its packets walked (header lengths), the
// count of packets and draws, and a hash of every header and every draw packet's words (the KYTY_PORT2 markers' payload,
// sealed later, is not hashed). A range whose packets do not walk to its end exactly is not registered.
// The GPU thread's side (graphics/guest_gpu/graphicsRun.cpp, the PM4 walk, before any draw path): at a draw packet
// inside a registered range, the first time the walk enters it, the range is walked again from the GPU thread's copy
// of the buffer and its signature compared (it must also lie inside the buffer being walked). In an ON window, a draw
// packet of a verified range is passed over: not translated, not recorded, nothing drawn. Every other packet of the
// range (registers, user data, index state, markers, labels, events, waits) runs as always, so the state the range
// leaves is the walk's and nothing waits forever. Anything unknown (no range, a different signature, outside the
// buffer, a state pass, a malformed packet): the draw runs as always. The lookup and the verification run in both
// windows (the same cost ON and OFF: the paired difference is the skipped draws' work alone).
// Windows: 5 s each, ON OFF OFF ON ON OFF OFF ON ... (ABBA, as KYTY_P7_ALTERNA: another 5 s alternation, such as
// KYTY_PORT2=1's, falls half in each state). A "[techo]" line every 5 s and "[techo] RESUMEN" at exit: ranges and draws
// skipped, GPU thread ms (busy and submissions) and frame ms ON vs OFF, AHORRO MEDIDO (mean OFF - mean ON) and the
// paired estimate (median of adjacent ON/OFF window pairs), and the decision against 1.5 ms.
// In ON windows the shadows and the prepass depth are missing: expected (measurement only).
// Off: one static load per packet of the walk, per hook call and per flip.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>
#include <x86intrin.h>

namespace F5p1Techo {

[[nodiscard]] inline bool Enabled() {
	const char* text = std::getenv("KYTY_F5P1_TECHO");
	if (text == nullptr || text[0] != '1' || text[1] != '\0') return false;
	const char* render1 = std::getenv("KYTY_RENDER1");
	if (render1 != nullptr && render1[0] != '\0' && !(render1[0] == '0' && render1[1] == '\0')) {
		std::printf("[techo] KYTY_F5P1_TECHO=1 DESACTIVADO: KYTY_RENDER1=%s usa el mismo gancho (quitalo para medir el techo)\n",
		            render1);
		std::fflush(stdout);
		return false;
	}
	std::printf("[techo] KYTY_F5P1_TECHO=1: ventanas de 5 s ON/OFF/OFF/ON; en ON el hilo GPU salta los draws de las pasadas "
	            "0/1 (ad8bc0: cascadas + prepass). SOLO MEDIDA: en ON faltan sombras y prepass\n");
	std::fflush(stdout);
	return true;
}
[[nodiscard]] inline bool On() {
	static const bool on = Enabled();
	return on;
}

inline std::atomic<bool> g_window_on {true}; // the window running now (flipped at the [techo] line)

constexpr uint32_t PassMask      = 0x3u;     // passes 0 and 1
constexpr size_t   MaxRecords    = 1u << 14u;
constexpr uint64_t MaxRangeBytes = 8u << 20u;
constexpr uint64_t MaxAgeMs      = 500;  // (the walk reaches a range well within this)
constexpr uint64_t SweepEveryMs  = 100;

struct Signature {
	uint32_t packets = 0, draws = 0;
	uint64_t hash    = 0;
	bool     ok      = false;
};

[[nodiscard]] inline bool IsDrawOpcode(uint32_t op) {
	// DRAW_INDIRECT, DRAW_INDEX_INDIRECT, DRAW_INDEX_2, DRAW_INDIRECT_MULTI, DRAW_INDEX_AUTO, DRAW_INDEX_OFFSET_2,
	// DRAW_INDEX_INDIRECT_MULTI
	return op == 0x24u || op == 0x25u || op == 0x27u || op == 0x2cu || op == 0x2du || op == 0x35u || op == 0x38u;
}
[[nodiscard]] inline uint64_t Mix(uint64_t h, uint64_t v) {
	h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
	return h * 0xff51afd7ed558ccdull;
}
// The packets of [p, end): every header and every draw packet's words. Not ok if a packet is not type 2 / 3, or does
// not end exactly at `end`.
[[nodiscard]] inline Signature Sign(const uint32_t* p, const uint32_t* end) {
	Signature s;
	uint64_t  h = 0x84222325cbf29ce4ull;
	while (p < end) {
		const uint32_t header = p[0];
		uint32_t       len    = 1;
		if (header != 0x80000000u) {
			if ((header >> 30u) != 3u) return s;
			len = ((header >> 16u) & 0x3fffu) + 2u;
			if (len > static_cast<uint64_t>(end - p)) return s;
			if (IsDrawOpcode((header >> 8u) & 0xffu)) {
				++s.draws;
				for (uint32_t i = 1; i < len; ++i) h = Mix(h, p[i]);
			}
		}
		h = Mix(h, header);
		++s.packets;
		p += len;
	}
	s.hash = Mix(h, (uint64_t {s.packets} << 32u) | s.draws);
	s.ok   = p == end;
	return s;
}

struct Record {
	uintptr_t end     = 0;
	uint64_t  born_ms = 0;
	uint32_t  pass    = 0;
	Signature sig;
};

// Counters (relaxed; read and cleared at each line).
enum Counter : uint32_t {
	HookCalls,       // calls of 0xad8bc0 (all passes)
	HookPass01,      // ... of pass 0 / 1
	Registered,      // pass 0/1 ranges registered
	RegSplit,        // not registered: its packets went to two blocks (or no command buffer)
	RegBad,          // not registered: its packets do not walk to its end, too big, or no draws
	RegFull,         // not registered: too many ranges held
	Stale,           // ranges dropped (overwritten by a newer one, or older than 0.5 s)
	Entered,         // ranges entered by the walk (first draw of the range in a walk)
	Verified,        // ... whose signature matched
	VerifyFailed,    // ... whose signature did not match (the draws ran as always)
	VerifyFailedOld, // (f5b) ... of them, registered more than 50 ms before (older than ~3 frames: memory used again)
	OutsideBuffer,   // ... not inside the buffer being walked (the draws ran as always)
	DrawsInRange,    // draw packets of verified ranges the walk reached (both windows)
	DrawsSkipped,    // ... passed over (ON windows)
	RangesSkipped,   // verified ranges entered in an ON window
	HookCycles,      // the hook's own work (game threads)
	ProbeCycles,     // the walk's lookups and verifications (GPU thread)
	Counters
};
inline std::atomic<uint64_t> g_count[Counters] {};
inline void Add(Counter c, uint64_t n = 1) { g_count[c].fetch_add(n, std::memory_order_relaxed); }

inline std::mutex                  g_mutex;
inline std::map<uintptr_t, Record> g_ranges;           // begin -> record (under g_mutex)
inline std::atomic<uint64_t>       g_erase_epoch {0}; // moves with every erase (under g_mutex)

[[nodiscard]] inline uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// --- the game's side (the hook of 0xad8bc0, after the game's function) --------------------------------------------
inline void Register(uint64_t pass_desc, uintptr_t begin, uintptr_t end, bool split) {
	if (!On()) return;
	const uint64_t t0 = __rdtsc();
	Add(HookCalls);
	const uint32_t pass = pass_desc != 0 ? *reinterpret_cast<const uint32_t*>(pass_desc + 0x20) : UINT32_MAX;
	if (pass >= 32 || ((PassMask >> pass) & 1u) == 0) {
		Add(HookCycles, __rdtsc() - t0);
		return;
	}
	Add(HookPass01);
	if (split || begin == 0 || end <= begin) {
		Add(RegSplit);
		Add(HookCycles, __rdtsc() - t0);
		return;
	}
	Signature sig;
	if (end - begin <= MaxRangeBytes && ((end - begin) & 3u) == 0)
		sig = Sign(reinterpret_cast<const uint32_t*>(begin), reinterpret_cast<const uint32_t*>(end));
	if (!sig.ok || sig.draws == 0) {
		Add(RegBad);
		Add(HookCycles, __rdtsc() - t0);
		return;
	}
	const uint64_t now = NowMs();
	{
		std::lock_guard lock(g_mutex);
		bool            erased = false;
		// Older ranges this one overlaps (the command buffer's memory used again).
		auto it = g_ranges.lower_bound(begin);
		if (it != g_ranges.begin() && std::prev(it)->second.end > begin) --it;
		while (it != g_ranges.end() && it->first < end) {
			it     = g_ranges.erase(it);
			erased = true;
			Add(Stale);
		}
		// Old ones (the walk passed them long ago), at most every 100 ms.
		static uint64_t last_sweep = 0;
		if (g_ranges.size() >= 1024 && now - last_sweep >= SweepEveryMs) {
			last_sweep = now;
			for (auto o = g_ranges.begin(); o != g_ranges.end();) {
				if (now - o->second.born_ms > MaxAgeMs) {
					o      = g_ranges.erase(o);
					erased = true;
					Add(Stale);
				} else {
					++o;
				}
			}
		}
		if (erased) g_erase_epoch.fetch_add(1, std::memory_order_release);
		if (g_ranges.size() >= MaxRecords) {
			Add(RegFull);
		} else {
			g_ranges.emplace(begin, Record {end, now, pass, sig});
			Add(Registered);
		}
	}
	Add(HookCycles, __rdtsc() - t0);
}

// --- the GPU thread's side (the PM4 walk) -------------------------------------------------------------------------
// At a packet of the walk (not a state pass): the number of dwords to pass over (the draw is skipped), or 0 (it runs
// as always). `buffer`, `buffer_dw`: the buffer being walked.
[[nodiscard]] inline uint32_t SkipDraw(uint32_t opcode, const uint32_t* packet, uint32_t remaining_dw,
                                       const uint32_t* buffer, uint32_t buffer_dw) {
	if (!IsDrawOpcode(opcode)) return 0;
	const uint64_t t0 = __rdtsc();
	struct Cache {
		uintptr_t begin = 0, end = 0, last = 0;
		uint64_t  epoch = UINT64_MAX;
		bool      ok    = false;
	};
	thread_local Cache c;
	const auto at    = reinterpret_cast<uintptr_t>(packet);
	const auto epoch = g_erase_epoch.load(std::memory_order_acquire);
	// The same walk inside the same range: what was decided when it entered.
	if (!(c.epoch == epoch && at >= c.begin && at < c.end && at > c.last)) {
		c = {};
		std::lock_guard lock(g_mutex);
		auto            it = g_ranges.upper_bound(at);
		if (it != g_ranges.begin() && at < std::prev(it)->second.end) {
			--it;
			const Record& r = it->second;
			c.begin         = it->first;
			c.end           = r.end;
			c.epoch         = g_erase_epoch.load(std::memory_order_relaxed);
			Add(Entered);
			const auto b0 = reinterpret_cast<uintptr_t>(buffer);
			const auto b1 = b0 + uint64_t {buffer_dw} * 4u;
			if (c.begin < b0 || c.end > b1) {
				Add(OutsideBuffer);
			} else {
				const Signature s = Sign(reinterpret_cast<const uint32_t*>(c.begin), reinterpret_cast<const uint32_t*>(c.end));
				c.ok = s.ok && s.packets == r.sig.packets && s.draws == r.sig.draws && s.hash == r.sig.hash;
				Add(c.ok ? Verified : VerifyFailed);
				if (!c.ok && NowMs() - r.born_ms > 50) Add(VerifyFailedOld); // (f5b: the stale-registration hypothesis)
				if (c.ok && g_window_on.load(std::memory_order_relaxed)) Add(RangesSkipped);
			}
		} else {
			c.epoch = UINT64_MAX; // (no range: looked up again at the next draw)
		}
	}
	c.last         = at;
	uint32_t skip  = 0;
	if (c.ok) {
		Add(DrawsInRange);
		const uint32_t len = ((packet[0] >> 16u) & 0x3fffu) + 2u;
		if (g_window_on.load(std::memory_order_relaxed) && (packet[0] >> 30u) == 3u && len <= remaining_dw &&
		    at + uint64_t {len} * 4u <= c.end) {
			skip = len;
			Add(DrawsSkipped);
		}
	}
	Add(ProbeCycles, __rdtsc() - t0);
	return skip;
}

// --- the report (render thread, at the flip) ----------------------------------------------------------------------
struct Window {
	bool   on = false, valid = false;
	double frames = 0, frame_ms = 0, busy_ms = 0, sub_ms = 0;
};
inline std::mutex          g_report_mutex;
inline std::vector<Window> g_windows; // (under g_report_mutex)
inline uint64_t            g_total[Counters] {};

struct Paired {
	size_t n = 0;
	double median = 0, mean = 0, half95 = 0;
};
// Pairs (2k, 2k+1) of the ABBA sequence: one ON and one OFF each, adjacent. `field`: OFF - ON of that measure.
[[nodiscard]] inline Paired Pairs(const std::vector<Window>& w, double Window::*field) {
	std::vector<double> d;
	for (size_t k = 0; k + 1 < w.size(); k += 2) {
		const Window& a = w[k];
		const Window& b = w[k + 1];
		if (!a.valid || !b.valid || a.on == b.on) continue;
		const Window& on  = a.on ? a : b;
		const Window& off = a.on ? b : a;
		d.push_back(off.*field - on.*field);
	}
	Paired p;
	p.n = d.size();
	if (d.empty()) return p;
	std::sort(d.begin(), d.end());
	p.median = d.size() % 2 ? d[d.size() / 2] : (d[d.size() / 2 - 1] + d[d.size() / 2]) / 2.0;
	double sum = 0;
	for (double x : d) sum += x;
	p.mean = sum / static_cast<double>(d.size());
	if (d.size() > 1) {
		double var = 0;
		for (double x : d) var += (x - p.mean) * (x - p.mean);
		var /= static_cast<double>(d.size() - 1);
		p.half95 = 1.96 * std::sqrt(var / static_cast<double>(d.size()));
	}
	return p;
}

inline void PrintSummary(const char* title, bool final_line) {
	std::lock_guard lock(g_report_mutex);
	double          acc[2][5] {}; // OFF, ON: windows, frames, frame ms, busy ms, sub ms (window means)
	for (const auto& w : g_windows) {
		if (!w.valid) continue;
		auto& a = acc[w.on ? 1 : 0];
		a[0] += 1;
		a[1] += w.frames;
		a[2] += w.frame_ms;
		a[3] += w.busy_ms;
		a[4] += w.sub_ms;
	}
	const auto& t = g_total;
	std::printf("[techo] %s: gancho %llu llamadas (p0/p1 %llu), rangos registrados %llu, partidos %llu, malos %llu, llenos %llu | "
	            "rangos entrados %llu, verificados %llu, firma distinta %llu (de rangos de mas de 50 ms %llu), fuera de bloque %llu | rangos saltados %llu, "
	            "draws en rango %llu, draws saltados %llu",
	            title, static_cast<unsigned long long>(t[HookCalls]), static_cast<unsigned long long>(t[HookPass01]),
	            static_cast<unsigned long long>(t[Registered]), static_cast<unsigned long long>(t[RegSplit]),
	            static_cast<unsigned long long>(t[RegBad]), static_cast<unsigned long long>(t[RegFull]),
	            static_cast<unsigned long long>(t[Entered]), static_cast<unsigned long long>(t[Verified]),
	            static_cast<unsigned long long>(t[VerifyFailed]), static_cast<unsigned long long>(t[VerifyFailedOld]),
	            static_cast<unsigned long long>(t[OutsideBuffer]),
	            static_cast<unsigned long long>(t[RangesSkipped]), static_cast<unsigned long long>(t[DrawsInRange]),
	            static_cast<unsigned long long>(t[DrawsSkipped]));
	if (acc[0][0] == 0 || acc[1][0] == 0) {
		std::printf(" | faltan ventanas ON y OFF validas (ON %.0f, OFF %.0f): sin AHORRO MEDIDO todavia\n", acc[1][0], acc[0][0]);
		std::fflush(stdout);
		return;
	}
	const auto m = [&](int w, int i) { return acc[w][i] / acc[w][0]; };
	const Paired pb = Pairs(g_windows, &Window::busy_ms);
	const Paired ps = Pairs(g_windows, &Window::sub_ms);
	const Paired pf = Pairs(g_windows, &Window::frame_ms);
	const bool   has_sub = m(0, 4) > 0 || m(1, 4) > 0;
	// The decision's measure: the GPU thread's submissions (KYTY_RENDER_STATS=1), else its busy time.
	const Paired& pd   = has_sub ? ps : pb;
	std::printf(" | ventanas OFF %.0f / ON %.0f | hilo GPU ocupado %.2f -> %.2f ms/frame, submissions %.2f -> %.2f ms/frame, "
	            "frame %.2f -> %.2f ms (OFF -> ON) | AHORRO MEDIDO hilo GPU %.3f ms/frame (ocupado), %.3f ms/frame "
	            "(submissions), frame %.3f ms | EMPAREJADO (%zu pares) ocupado mediana %.3f media %.3f +-%.3f, submissions "
	            "mediana %.3f media %.3f +-%.3f, frame mediana %.3f media %.3f +-%.3f ms",
	            acc[0][0], acc[1][0], m(0, 3), m(1, 3), m(0, 4), m(1, 4), m(0, 2), m(1, 2), m(0, 3) - m(1, 3),
	            m(0, 4) - m(1, 4), m(0, 2) - m(1, 2), pb.n, pb.median, pb.mean, pb.half95, ps.median, ps.mean, ps.half95,
	            pf.median, pf.mean, pf.half95);
	if (pd.n == 0) {
		std::printf(" | DECISION: sin pares todavia\n");
	} else {
		std::printf(" | DECISION (%s, mediana emparejada %.3f ms, %zu pares%s): %s\n", has_sub ? "submissions" : "hilo GPU ocupado",
		            pd.median, pd.n, pd.n < 6 ? ", POCOS PARES: medir mas" : "",
		            pd.median < 1.5 ? "<1,5 ms: piloto por pasadas CERRADO (F5 solo tiene sentido completo)"
		                            : ">=1,5 ms: RECALCULAR la variante completa con r medido");
	}
	(void)final_line;
	std::fflush(stdout);
}

inline void AtExit() { PrintSummary("RESUMEN FINAL", true); }

// `idle_ns`: LiveControl::g_render_idle_ns (the GPU thread's waits for guest work); `sub_cycles`: the GPU thread's
// submissions (LiveCensus::g_kind_cycles[Submission], 0 without KYTY_RENDER_STATS).
inline void Report(int64_t now_ns, int64_t idle_ns, uint64_t sub_cycles) {
	if (!On()) return;
	static int64_t  last_ns = 0, last_idle = 0;
	static uint64_t last_tsc = 0, last_sub = 0, frames = 0, window = 0, lines = 0;
	++frames;
	const uint64_t tsc = __rdtsc();
	if (last_ns == 0) {
		last_ns   = now_ns;
		last_idle = idle_ns;
		last_tsc  = tsc;
		last_sub  = sub_cycles;
		frames    = 0;
		std::atexit(AtExit);
		return;
	}
	if (now_ns - last_ns < 5'000'000'000 || frames == 0 || tsc <= last_tsc) return;
	const double n        = static_cast<double>(frames);
	const double wall_ms  = static_cast<double>(now_ns - last_ns) / 1e6;
	const double ns_cycle = wall_ms * 1e6 / static_cast<double>(tsc - last_tsc);
	Window       w;
	w.on       = g_window_on.load(std::memory_order_relaxed);
	w.frames   = n;
	w.frame_ms = wall_ms / n;
	w.busy_ms  = (wall_ms - static_cast<double>(idle_ns - last_idle) / 1e6) / n;
	w.sub_ms   = static_cast<double>(sub_cycles - last_sub) * ns_cycle / 1e6 / n;
	w.valid    = frames >= 20 && w.frame_ms < 100.0; // (loading screens and stalls out)
	uint64_t d[Counters];
	for (uint32_t i = 0; i < Counters; ++i) {
		d[i] = g_count[i].exchange(0, std::memory_order_relaxed);
		g_total[i] += d[i];
	}
	{
		std::lock_guard lock(g_report_mutex);
		g_windows.push_back(w);
	}
	std::printf("[techo] %.1f s, %.0f frames, %s%s | por frame: rangos p0/p1 registrados %.1f, entrados %.1f (firma distinta %.1f, "
	            "fuera de bloque %.1f), saltados %.1f | draws en rango %.1f, saltados %.1f | hilo GPU ocupado %.2f ms, "
	            "submissions %.2f ms, frame %.2f ms | coste sonda: gancho %.3f ms (hilos del juego), hilo GPU %.3f ms\n",
	            wall_ms / 1e3, n, w.on ? "ON" : "OFF", w.valid ? "" : " (no valida)", d[Registered] / n, d[Entered] / n,
	            d[VerifyFailed] / n, d[OutsideBuffer] / n, d[RangesSkipped] / n, d[DrawsInRange] / n, d[DrawsSkipped] / n,
	            w.busy_ms, w.sub_ms, w.frame_ms, static_cast<double>(d[HookCycles]) * ns_cycle / 1e6 / n,
	            static_cast<double>(d[ProbeCycles]) * ns_cycle / 1e6 / n);
	if (++lines % 6 == 0) PrintSummary("acumulado", false);
	std::fflush(stdout);
	last_ns   = now_ns;
	last_idle = idle_ns;
	last_tsc  = tsc;
	last_sub  = sub_cycles;
	frames    = 0;
	// ON OFF OFF ON ON OFF OFF ON ... (window counts the windows closed).
	++window;
	g_window_on.store(window % 4 == 0 || window % 4 == 3, std::memory_order_relaxed);
}

} // namespace F5p1Techo

#endif // EMULATOR_SRC_LOCAL_F5P1_TECHO_H_
