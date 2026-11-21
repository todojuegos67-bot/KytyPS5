// (f5b) KYTY_NATIVE_TECHO=<pass>,<pass>,...: the per-pass ceiling, all listed passes in one session.
//
// What it measures: how much GPU-thread time the emulator spends on the draws / dispatches of each pass, i.e. the most
// a native version of that pass can take off the GPU thread (its own native recording costs come on top: f5 §4).
// How: the frame map's scope stacks give each packet's pass (pass-cut.h); in an ON window the GPU thread's walk passes
// over the draw and dispatch packets whose innermost open pass is that window's pass (a name with '*' at the end: open
// at any depth). Every other packet of the pass runs as always: registers, user data, index state, labels, events,
// waits, so the state its range leaves is the walk's and nothing waits forever. In its ON window the pass's image is
// missing: expected, test folder only.
// Windows (KYTY_NATIVE_TECHO_MS, default 5000 ms): OFF, A, B, OFF, C, D, OFF, ... (each ON window next to one OFF
// window, the passes in turn). Pairs: each ON window with its OFF neighbour (the one before for the first of the two,
// the one after for the second). Per pass, OFF - ON of: the GPU thread's submissions (KYTY_RENDER_STATS), its busy
// time, its time in slices (the map), the frame, and the pass's own exclusive time (the least noisy: the scene changes
// both windows alike; inclusive time for a name with '*'). Median, mean and +-95% of the pairs. Lines "[f5t]" every window, "[f5t] RESUMEN" every 60 s
// and at exit.
#include "native/native-render.h"
#include "native/pass-cut.h"
#include "native/pass-registry.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <x86intrin.h>

namespace NativeRender::Techo {
namespace {

constexpr uint32_t MaxTargets = 32;

struct Target {
	std::string           name;      // as written
	bool                  inclusive = false;
	std::vector<uint32_t> ids;       // registry ids with this name (resolved as names appear)
};

struct Window {
	int32_t target = -1; // -1: OFF
	bool    valid  = false;
	double  frames = 0, frame_ms = 0, busy_ms = 0, sub_ms = 0, slice_ms = 0;
	double  excl_ms[MaxTargets] {};   // every target's exclusive ms/frame in this window
	double  in_draws[MaxTargets] {};  // draw + dispatch packets of the target's pass the walk reached, per frame
	double  skipped = 0;              // packets passed over per frame (ON)
};

struct State {
	bool                configured = false;
	std::vector<Target> targets;
	int16_t             target_of[MaxPasses];
	uint32_t            resolved = 0;
	int64_t             window_ns = 5'000'000'000LL;
	// the window running
	int32_t  current = -1;
	uint64_t index   = 0;
	bool     started = false;
	int64_t  start_ns = 0, start_idle = 0, start_slice = 0;
	uint64_t start_tsc = 0, start_sub = 0, frames = 0;
	int64_t  start_excl[MaxTargets] {};
	uint64_t in_draws[MaxTargets] {}, skipped = 0;
	// history
	std::vector<Window> windows;
	int64_t             last_summary = 0;
	bool                exit_hooked  = false;
};
State g;

bool IsDrawOrDispatch(uint32_t op) {
	// DRAW_INDIRECT, DRAW_INDEX_INDIRECT, DRAW_INDEX_2, DRAW_INDIRECT_MULTI, DRAW_INDEX_AUTO, DRAW_INDEX_OFFSET_2,
	// DRAW_INDEX_INDIRECT_MULTI; DISPATCH_DIRECT, DISPATCH_INDIRECT
	return op == 0x24u || op == 0x25u || op == 0x27u || op == 0x2cu || op == 0x2du || op == 0x35u || op == 0x38u ||
	       op == 0x15u || op == 0x16u;
}

void Resolve() {
	const uint32_t n = PassRegistry::Count();
	for (uint32_t id = g.resolved; id < n && id < MaxPasses; ++id) {
		g.target_of[id] = -1;
		for (size_t t = 0; t < g.targets.size(); ++t)
			if (PassRegistry::SameName(PassRegistry::Name(id), g.targets[t].name.c_str())) {
				g.target_of[id] = static_cast<int16_t>(t);
				g.targets[t].ids.push_back(id);
				std::printf("[f5t] pasada \"%s\" -> id %u%s\n", g.targets[t].name.c_str(), id,
				            g.targets[t].inclusive ? " (inclusiva: abierta a cualquier profundidad)" : "");
				std::fflush(stdout);
				break;
			}
	}
	g.resolved = n;
}

// OFF, A, B, OFF, C, D, ...: window w's target (-1 OFF).
int32_t TargetOfWindow(uint64_t w) {
	if (g.targets.empty() || w % 3 == 0) return -1;
	return static_cast<int32_t>((2 * (w / 3) + (w % 3 - 1)) % g.targets.size());
}

struct Paired {
	size_t n = 0;
	double median = 0, mean = 0, half95 = 0;
};
Paired Stats(std::vector<double> d) {
	Paired p;
	p.n = d.size();
	if (d.empty()) return p;
	std::sort(d.begin(), d.end());
	p.median = d.size() % 2 ? d[d.size() / 2] : (d[d.size() / 2 - 1] + d[d.size() / 2]) / 2.0;
	double sum = 0;
	for (const double x: d) sum += x;
	p.mean = sum / static_cast<double>(d.size());
	if (d.size() > 1) {
		double var = 0;
		for (const double x: d) var += (x - p.mean) * (x - p.mean);
		var /= static_cast<double>(d.size() - 1);
		p.half95 = 1.96 * std::sqrt(var / static_cast<double>(d.size()));
	}
	return p;
}

void Summary(const char* title) {
	const auto& w = g.windows;
	std::printf("[f5t] %s: %zu ventanas (%.1f s cada una); por pasada, OFF - ON emparejado (ventana ON con su OFF vecina):\n",
	            title, w.size(), static_cast<double>(g.window_ns) / 1e9);
	struct Row {
		size_t t;
		Paired sub, busy, slice, frame, local;
		double excl_off = 0, skipped = 0;
	};
	std::vector<Row> rows;
	for (size_t t = 0; t < g.targets.size(); ++t) {
		std::vector<double> sub, busy, slice, frame, local;
		double              excl_off = 0, skipped = 0;
		for (size_t k = 0; k < w.size(); ++k) {
			if (w[k].target != static_cast<int32_t>(t) || !w[k].valid) continue;
			const size_t o = k % 3 == 1 ? k - 1 : k + 1;
			if (o >= w.size() || w[o].target != -1 || !w[o].valid) continue;
			sub.push_back(w[o].sub_ms - w[k].sub_ms);
			busy.push_back(w[o].busy_ms - w[k].busy_ms);
			slice.push_back(w[o].slice_ms - w[k].slice_ms);
			frame.push_back(w[o].frame_ms - w[k].frame_ms);
			local.push_back(w[o].excl_ms[t] - w[k].excl_ms[t]);
			excl_off += w[o].excl_ms[t];
			skipped += w[k].skipped;
		}
		Row r {t, Stats(sub), Stats(busy), Stats(slice), Stats(frame), Stats(local)};
		if (!local.empty()) {
			r.excl_off = excl_off / static_cast<double>(local.size());
			r.skipped  = skipped / static_cast<double>(local.size());
		}
		rows.push_back(r);
	}
	const bool has_sub = std::any_of(w.begin(), w.end(), [](const Window& x) { return x.sub_ms > 0; });
	std::sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) {
		return (has_sub ? a.sub.median : a.busy.median) > (has_sub ? b.sub.median : b.busy.median);
	});
	uint32_t rank = 0;
	for (const auto& r: rows) {
		const Paired& main = has_sub ? r.sub : r.busy;
		std::printf("[f5t] %s #%u \"%s\" pares=%zu | submissions %.3f (media %.3f +-%.3f) | ocupado %.3f (+-%.3f) | slices %.3f "
		            "(+-%.3f) | frame %.3f (+-%.3f) | propia %.3f (+-%.3f) de %.3f ms/frame OFF | saltados %.1f/frame -> %.2f "
		            "us/paquete | r=%.2f | %s\n",
		            title, ++rank, g.targets[r.t].name.c_str(), main.n, r.sub.median, r.sub.mean, r.sub.half95,
		            r.busy.median, r.busy.half95, r.slice.median, r.slice.half95, r.frame.median, r.frame.half95,
		            r.local.median, r.local.half95, r.excl_off, r.skipped,
		            r.skipped > 0 ? main.median * 1000.0 / r.skipped : 0.0,
		            r.excl_off > 0 ? r.slice.median / r.excl_off : 0.0,
		            main.n < 3                       ? "POCOS PARES: medir mas"
		            : main.median > main.half95      ? "AHORRO MEDIDO (fuera del ruido)"
		            : r.local.median > r.local.half95 ? "solo la medida propia sale del ruido"
		                                              : "dentro del ruido");
	}
	if (!has_sub)
		std::printf("[f5t] %s: sin KYTY_RENDER_STATS=1 no hay submissions; el orden va por hilo GPU ocupado\n", title);
	std::fflush(stdout);
}

void AtExit() {
	if (g.configured && !g.windows.empty()) Summary("RESUMEN FINAL");
}

bool ReadConfig() {
	const char* v = std::getenv("KYTY_NATIVE_TECHO");
	if (v == nullptr || v[0] == '\0' || (v[0] == '0' && v[1] == '\0')) return false;
	std::fill(std::begin(g.target_of), std::end(g.target_of), static_cast<int16_t>(-1));
	const std::string s(v);
	size_t            at = 0;
	while (at <= s.size()) {
		size_t end = s.find(',', at);
		if (end == std::string::npos) end = s.size();
		std::string item = s.substr(at, end - at);
		while (!item.empty() && item.front() == ' ') item.erase(item.begin());
		while (!item.empty() && item.back() == ' ') item.pop_back();
		Target t;
		if (!item.empty() && item.back() == '*') {
			t.inclusive = true;
			item.pop_back();
		}
		t.name = item;
		if (!t.name.empty() && g.targets.size() < MaxTargets) g.targets.push_back(t);
		at = end + 1;
	}
	if (const char* ms = std::getenv("KYTY_NATIVE_TECHO_MS"); ms != nullptr && std::atoi(ms) >= 1000)
		g.window_ns = int64_t {std::atoi(ms)} * 1'000'000;
	if (g.targets.empty()) return false;
	std::string list;
	for (const auto& t: g.targets) list += (list.empty() ? "" : ", ") + t.name + (t.inclusive ? "*" : "");
	std::printf("[f5t] KYTY_NATIVE_TECHO: %zu pasadas (%s); ventanas de %.1f s OFF, A, B, OFF, C, D...: en la ventana de una "
	            "pasada el hilo GPU salta sus draws y dispatches (lo demas de su rango corre). SOLO MEDIDA: en su ventana falta "
	            "la imagen de esa pasada. Un ciclo completo: %.0f s\n",
	            g.targets.size(), list.c_str(), static_cast<double>(g.window_ns) / 1e9,
	            static_cast<double>(g.window_ns) / 1e9 * static_cast<double>((g.targets.size() + 1) / 2 * 3));
	std::fflush(stdout);
	return true;
}

} // namespace

bool Configured() noexcept {
	static const bool on = [] {
		g.configured = ReadConfig();
		return g.configured;
	}();
	return on;
}

bool Skip(uint32_t opcode, uint32_t id) noexcept {
	if (!IsDrawOrDispatch(opcode)) return false;
	if (PassRegistry::Count() != g.resolved) Resolve();
	const int32_t t = id < MaxPasses ? g.target_of[id] : -1;
	bool          hit = g.current >= 0 && t == g.current;
	if (t >= 0 && !g.targets[static_cast<size_t>(t)].inclusive) ++g.in_draws[t];
	// Inclusive targets: open at any depth (counted in both windows; the window's one is passed over).
	for (size_t k = 0; k < g.targets.size(); ++k) {
		if (!g.targets[k].inclusive) continue;
		for (const uint32_t i: g.targets[k].ids)
			if (Cut::IsOpen(i)) {
				++g.in_draws[k];
				hit |= static_cast<int32_t>(k) == g.current;
				break;
			}
	}
	if (hit) ++g.skipped;
	return hit;
}

void Flip(int64_t now_ns, int64_t idle_ns, uint64_t sub_cycles) {
	if (PassRegistry::Count() != g.resolved) Resolve();
	const uint64_t tsc = __rdtsc();
	const auto     excl_of = [](size_t t) {
		int64_t sum = 0;
		for (const uint32_t id: g.targets[t].ids) sum += g.targets[t].inclusive ? Cut::InclNs(id) : Cut::ExclNs(id);
		return sum;
	};
	if (!g.started) {
		g.started = true;
		if (!g.exit_hooked) {
			g.exit_hooked = true;
			std::atexit(AtExit);
		}
	} else {
		++g.frames;
		if (now_ns - g.start_ns < g.window_ns || g.frames == 0 || tsc <= g.start_tsc) return;
		const double n       = static_cast<double>(g.frames);
		const double wall_ms = static_cast<double>(now_ns - g.start_ns) / 1e6;
		const double ns_tsc  = wall_ms * 1e6 / static_cast<double>(tsc - g.start_tsc);
		Window       w;
		w.target   = g.current;
		w.frames   = n;
		w.frame_ms = wall_ms / n;
		w.busy_ms  = (wall_ms - static_cast<double>(idle_ns - g.start_idle) / 1e6) / n;
		w.sub_ms   = static_cast<double>(sub_cycles - g.start_sub) * ns_tsc / 1e6 / n;
		w.slice_ms = static_cast<double>(Cut::SliceNs() - g.start_slice) / 1e6 / n;
		for (size_t t = 0; t < g.targets.size(); ++t) {
			w.excl_ms[t]  = static_cast<double>(excl_of(t) - g.start_excl[t]) / 1e6 / n;
			w.in_draws[t] = static_cast<double>(g.in_draws[t]) / n;
		}
		w.skipped = static_cast<double>(g.skipped) / n;
		w.valid   = g.frames >= 20 && w.frame_ms < 100.0; // (loading screens and stalls out)
		g.windows.push_back(w);
		if (w.target >= 0) {
			const auto t = static_cast<size_t>(w.target);
			std::printf("[f5t] ventana %llu ON \"%s\"%s: %.1f s, %.0f frames | frame %.2f ms, hilo GPU ocupado %.2f, submissions %.2f, "
			            "slices %.2f ms/frame | la pasada: propia %.3f ms/frame, draws+dispatches %.1f/frame, saltados %.1f/frame\n",
			            static_cast<unsigned long long>(g.index), g.targets[t].name.c_str(), w.valid ? "" : " (no valida)",
			            wall_ms / 1e3, n, w.frame_ms, w.busy_ms, w.sub_ms, w.slice_ms, w.excl_ms[t], w.in_draws[t], w.skipped);
		} else {
			std::printf("[f5t] ventana %llu OFF%s: %.1f s, %.0f frames | frame %.2f ms, hilo GPU ocupado %.2f, submissions %.2f, "
			            "slices %.2f ms/frame | propia por pasada:",
			            static_cast<unsigned long long>(g.index), w.valid ? "" : " (no valida)", wall_ms / 1e3, n, w.frame_ms,
			            w.busy_ms, w.sub_ms, w.slice_ms);
			for (size_t t = 0; t < g.targets.size(); ++t)
				std::printf(" %s=%.3f(%.0f)", g.targets[t].name.c_str(), w.excl_ms[t], w.in_draws[t]);
			std::printf("\n");
		}
		std::fflush(stdout);
		if (now_ns - g.last_summary >= 60'000'000'000LL) {
			Summary("acumulado");
			g.last_summary = now_ns;
		}
		++g.index;
	}
	// The next window.
	g.current     = TargetOfWindow(g.index);
	g.start_ns    = now_ns;
	g.start_idle  = idle_ns;
	g.start_tsc   = tsc;
	g.start_sub   = sub_cycles;
	g.start_slice = Cut::SliceNs();
	g.frames      = 0;
	g.skipped     = 0;
	for (size_t t = 0; t < g.targets.size(); ++t) {
		g.start_excl[t] = excl_of(t);
		g.in_draws[t]   = 0;
	}
	if (g.last_summary == 0) g.last_summary = now_ns;
}

} // namespace NativeRender::Techo
