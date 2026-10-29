#ifndef KYTY_LOCAL_DYN_MULTIPLIER_H_
#define KYTY_LOCAL_DYN_MULTIPLIER_H_

// 10-22: the dynamic frame-generation multiplier shared by DLSS-G (frame-gen.cpp) and FSR3 MFG (fsr3-fg.cpp).
// Decided on the GPU's free render rate, not the rate seen under the current multiplier (generating more frames
// costs GPU time; "largest m with real * m <= target" read on that lowered rate spiralled x4 -> x6 at 9 real fps):
// - free[m]: the real rate last measured at multiplier m (EMA, not sitting on the target / m cap); the rate at m + 1
//   is estimated as 1 / (1 / free[m] + cost), cost = measured seconds per extra generated frame (from switches),
//   8% of the frame until one was measured.
// - Minimum real fps per multiplier (10-23): x2 20, x3 26, x4 30, x5/x6 36; never up when real (or the estimate) is below the
//   next one's, down after 2 windows in a row below the current one's; m only where target / m >= its minimum. A
//   window under 60% of the one before (hitch, loading) is ignored.
// - Up at once when real * (m + 1) <= 1.03 target (the game's own 56-60 cap keeps real), or after 5 s under 90% of
//   the target (a probe; the check below reverts it); otherwise on the estimate.
// - Unlocked (real above 60): real <= target / m; on that cap the lowest m whose free rate reaches target / m (down
//   one step when a measured free rate >= 1.1 target / (m - 1) for 2 windows, else stay; reverted when it shows < 97%
//   of the target, then not again for 30 s doubling to 4 min). Never above the target. Each level held >= 3 s.
// - Up one level at a time, only when it shows >= 5% more (min(real * m, target)); 2 windows after a step up, the
//   shown rate is checked against the one before: not >= 5% higher (or real under the minimum) -> back down and that
//   level blocked for 10 s. The level that reaches the target with real capped by <= 10% is preferred (60 real on
//   224: x4 at 56 real, kept when it shows >= 97% of the target).
// - The cap (real <= target / m) binds only when real * m would pass the target; when real sits on it for 2 windows
//   and the lower multiplier's free rate also reaches the target, one step down (no step up for 10 s).
// - 10-23 quality rule (KYTY_DYN_QUALITY=<fraction of the target>, default 0.7; 0 = off, the rules above only):
//   fewer generated frames when the GPU has headroom (better image, less input lag). Down one level when
//   real * (m - 1) >= q * target for 2 windows (x4 held at 56 on 224: 168 >= 157 -> x3, real let up to 74.7);
//   on the cap target / m for 3 windows without that, a probe one level down (reverted when real * (m - 1) shows
//   < q * target, then not again for 30 s doubling to 4 min). Up only when the shown rate is under (q - 0.05) *
//   target for 3 windows in a row (hysteresis: x3 at 54-60 real stays x3 on 224; x2 kept from 78 real).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace Kyty::Local {

struct DynMultiplier {
	static constexpr int kMax = 8;
	std::array<double, kMax + 1> free {};  // real fps measured at m (0 = unknown)
	std::array<int, kMax + 1>    age {};   // windows since free[m] was measured
	std::array<int, kMax + 1>    block {}; // windows left with level m blocked
	std::array<int, kMax + 1>    reverts {}; // step-up reverts of level m (each doubles its block)
	double cost       = 0.0;               // seconds per extra generated frame (0 = unknown)
	int    settle     = 0;                 // windows left before measuring after a switch
	int    check_from = 0;                 // after a step up: the level we came from (0 = none)
	double check_real = 0.0, check_shown = 0.0;
	int    pinned = 0, noup = 0, up_n = 0;
	double last_free = 0.0;                // the free estimate of the current level (log)
	double prev_real = 0.0;                // the window before (hitch test)
	int    down_from = 0, nodown = 0, nodown_len = 15, free_n = 0; // a step down off the cap under test; probes paused; windows free
	int    low_n = 0, short_n = 0;         // windows in a row under the level's minimum / under 90% of the target
	int    qdown_n = 0, qup_n = 0;         // quality rule: windows in a row a lower level fits / shown under (q - 0.05) T
	const char* reason = "start";          // 10-29: why the last switch was made (log)
	double next_est = 0.0;                 // 10-29: the free estimate of the level switched to (log)
	int    collapse_n = 0;                 // 10-29: windows in a row real far under what the level below predicts

	static double Quality() {
		static const double q = [] {
			const char* v = std::getenv("KYTY_DYN_QUALITY");
			const double r = v != nullptr && *v != '\0' ? std::atof(v) : 0.7;
			std::printf("[dyn] quality rule: %s (KYTY_DYN_QUALITY=%.2f of the target)\n", r > 0.0 ? "on" : "off", r);
			return std::clamp(r, 0.0, 1.0);
		}();
		return q;
	}

	// 10-23: lower minimums (x2 20, x3 26, x4 30, x5+ 36): with a heavy load (4K cinematic, ~37-48 real) x3 fell
	// to x2 under the old 40 and the screen showed 80-90 of 224, flipping x2/x3 every few seconds. The aim is the
	// monitor's rate; fewer generated frames only when the GPU has headroom (the quality rule).
	// KYTY_DYN_MIN_SCALE scales them (1 = these; 1.5 = the old ones).
	static double MinReal(int m) {
		static const double k = [] {
			const char* v = std::getenv("KYTY_DYN_MIN_SCALE");
			return v != nullptr && *v != '\0' ? std::clamp(std::atof(v), 0.25, 3.0) : 1.0;
		}();
		return k * (m <= 2 ? 20.0 : m == 3 ? 26.0 : m == 4 ? 30.0 : 36.0);
	}

	double Estimate(int m, double base_real, int base_m) const {
		if (free[m] > 0.0 && age[m] < 10) return free[m]; // (recent only: the scene moves on)
		const double c = cost > 0.0 ? cost : 0.08 / std::max(base_real, 1.0);
		return 1.0 / (1.0 / std::max(base_real, 1.0) + c * (m - base_m));
	}

	// Once a second. m: current multiplier (>= 2), real: this window's real fps, T: shown target, maxm: the largest
	// multiplier allowed now (cinematics pass 3). Returns the multiplier for the next window.
	// hold: a cinematic, held cutscene or movie (real frames held to 60/30): nothing learned or blocked, all state reset
	// so the levels and estimates start fresh when it ends.
	// free_real: the rate without our limiter when measured (FSR3), 0 when not (Reflex).
	int Window(int m, double real, double T, int maxm, bool hold = false, double free_real = 0.0) {
		if (hold) {
			*this = DynMultiplier {};
			return std::clamp(std::min(m, maxm), 2, kMax);
		}
		for (auto& b: block) b = std::max(b - 1, 0);
		for (auto& a: age) ++a;
		if (noup > 0) --noup;
		if (nodown > 0) --nodown;
		while (maxm > 2 && T / maxm < MinReal(maxm)) --maxm;
		maxm = std::clamp(maxm, 2, kMax);
		const double cap = T / m;
		const bool   on_cap = real >= 0.98 * cap;
		pinned = on_cap ? pinned + 1 : 0;
		if (settle > 0) { // (at least 3 s per level: the switch's window and the next are not decided on)
			--settle;
			prev_real = real;
			last_free = Estimate(m, real, m);
			return m;
		}
		// A hitch or loading (real under 60% of the window before): nothing learned or decided on it.
		const double before = prev_real;
		prev_real           = real;
		if (before > 0.0 && real < 0.6 * before) {
			last_free = Estimate(m, before, m);
			return m;
		}
		// Measure: an EMA of the free rate (a window on the cap is only a lower bound).
		if (!on_cap || free[m] < real) free[m] = (free[m] > 0.0 && age[m] < 30 && !on_cap) ? 0.5 * free[m] + 0.5 * real : real;
		age[m]    = 0;
		// A higher level's stored rate far under what this level measures now was learned in another scene: dropped.
		for (int k = m + 1; k <= kMax; ++k)
			if (free[k] > 0.0 && free[k] < 0.75 * real) free[k] = 0.0;
		last_free = free[m];
		low_n     = real < MinReal(m) ? low_n + 1 : 0;
		short_n   = std::min(real * m, T) < 0.9 * T ? short_n + 1 : 0;
		auto go = [&](int to, const char* why = "rule") {
			reason   = why;
			next_est = Estimate(to, real, m);
			settle  = 2;
			up_n    = 0;
			pinned  = 0;
			low_n   = 0;
			short_n = 0;
			qdown_n = 0;
			qup_n   = 0;
			return to;
		};
		// A step up under test: kept with >= 5% more shown (or >= 97% of the target) and real at the level's minimum.
		if (check_from != 0) {
			const int from = check_from;
			check_from     = 0;
			const double shown = std::min(real * m, T);
			if (real > 0.0 && check_real > 0.0 && real >= 0.7 * check_real) { // (a larger drop is not a frame's cost)
				const double c = (1.0 / real - 1.0 / check_real) / std::max(m - from, 1);
				if (c > 0.0) cost = cost > 0.0 ? 0.5 * cost + 0.5 * c : c;
			}
			if ((shown < 1.05 * check_shown && shown < 0.97 * T) || real < MinReal(m)) {
				// 10-23: each revert of the same level blocks it twice as long (30 s, 60, 120, up to 4 min): x4/x5 every
				// ~5 s changed the pacing each time (a hitch per switch, 20261023-205449).
				reverts[m] = std::min(reverts[m] + 1, 4);
				block[m]   = 30 << (reverts[m] - 1);
				free[m]  = 0.0;
				return go(from, "step up not kept (shows < 5% more or real under the level's minimum)");
			}
			reverts[m] = 0; // kept
		}
		// Down after 2 windows in a row below the level's minimum (to the largest lower level whose known rate meets its own).
		if (m > 2 && (low_n >= 2 || m > maxm)) {
			int to = std::min(m - 1, maxm);
			while (to > 2 && free[to] > 0.0 && age[to] < 30 && free[to] < MinReal(to)) --to;
			return go(to, "real under the level's minimum for 2 s");
		}
		// A step down off the cap under test: kept when it still shows >= 97% of the target, else back up and no
		// step-down probe for 30 s, doubling per failed probe up to 4 min (a game held at 60 is not probed often).
		if (down_from != 0) {
			const int from = down_from;
			down_from      = 0;
			const double keep = Quality() > 0.0 ? Quality() * T : 0.97 * T;
			if (std::min(real * m, T) < keep && from <= maxm) {
				std::printf("[dyn] step down to x%d reverted: shown %.1f < %.1f\n", m, std::min(real * m, T), keep);
				nodown_len = std::min(nodown_len * 2, 240);
				nodown     = nodown_len;
				return go(from, "step down reverted (shows under the keep rate)");
			}
			nodown_len = 15;
		}
		// 10-29: a real rate far under what the level below predicts for this one (free[m - 1] plus one generated
		// frame's cost) is the emulator's own pacing collapsing (log 20261029-032959: x4 with the game's flips held at
		// 56 fell 59 -> 36 real, then x5/x6 climbed on it), not the GPU: never a reason to go up. Back one level, and
		// that level is not blocked (the pacing, not the GPU, failed); the step up is retried only by the rules.
		{
			const double below = m > 2 && free[m - 1] > 0.0 && age[m - 1] < 30 ? free[m - 1] : 0.0;
			const double pred  = below > 0.0 ? 1.0 / (1.0 / below + (cost > 0.0 ? cost : 0.08 / below)) : 0.0;
			collapse_n         = pred > 0.0 && real < 0.8 * std::min(pred, cap) ? collapse_n + 1 : 0;
			if (collapse_n >= 2) {
				std::printf("[dyn] x%d collapse: real %.1f < 80%% of %.1f predicted from x%d's free %.1f (pacing, not the GPU)\n", m, real,
				            std::min(pred, cap), m - 1, below);
				collapse_n = 0;
				free[m]    = 0.0;
				noup       = 10;
				return go(m - 1, "real collapsed under the lower level's prediction (pacing)");
			}
		}
		// 10-23 quality rule: the lowest multiplier that still shows q * target, with hysteresis (see the top).
		if (const double q = Quality(); q > 0.0) {
			const double shown = std::min(real * m, T);
			qup_n              = shown < (q - 0.05) * T ? qup_n + 1 : 0;
			if (m > 2 && nodown == 0) {
				const bool fits = real * (m - 1) >= q * T;
				qdown_n         = fits ? qdown_n + 1 : 0;
				if (qdown_n >= 2) {
					std::printf("[dyn] headroom: x%d -> x%d (real %.1f, x%d shows %.1f >= %.1f)\n", m, m - 1, real, m - 1,
					            real * (m - 1), q * T);
					noup = 5;
					return go(m - 1, "headroom: one level less still shows q x target");
				}
				// 10-29: no probe when the level below is known (recent) not to reach the keep rate: x4 held at 56 on 224
				// with x3 measured at 60 (180 < 212.8) bounced x4 -> x3 -> x4 and often stayed at x3 showing 180.
				const bool below_short = free[m - 1] > 0.0 && age[m - 1] < 240 && free[m - 1] * (m - 1) < q * T;
				if (!fits && pinned >= 3 && !below_short) { // held on target / m: the game may run faster with fewer generated frames
					std::printf("[dyn] headroom probe: x%d -> x%d (real %.1f held at the cap %.1f)\n", m, m - 1, real, cap);
					noup      = 5;
					down_from = m;
					return go(m - 1, "headroom probe (held on the cap target / m)");
				}
			} else {
				qdown_n = 0;
			}
			const int up = m + 1;
			// 10-29: the smallest multiplier that reaches ~target: up only while the next level's free estimate meets
			// its own minimum; x(m + 1) at target / (m + 1) when the measured rate reaches it (60 real at x3 -> x4 at
			// 56). A step-up block from a pacing collapse does not hold a level whose estimate reaches target / up.
			const double est_up = Estimate(up, real, m);
			if (up <= maxm && block[up] > 0 && qup_n >= 3 && real >= 0.97 * T / up && est_up >= 0.97 * T / up) block[up] = 0;
			if (up <= maxm && block[up] == 0 && noup == 0 && qup_n >= 3 && real >= MinReal(up)) {
				std::printf("[dyn] x%d shows %.1f < %.1f for 3 s: up to x%d\n", m, shown, (q - 0.05) * T, up);
				check_from  = m;
				check_real  = real;
				check_shown = shown;
				return go(up, real >= 0.97 * T / up ? "shows < (q - 0.05) x target for 3 s; real reaches target / (m + 1)"
				                                    : "shows < (q - 0.05) x target for 3 s; real under target / (m + 1), more frames needed");
			}
			return m;
		}
		// On the cap (the game runs faster than target / m, unlocked above 60): the lowest m whose free rate reaches
		// target / m. With a measured free rate (FSR3's limiter) down when it is >= 1.05 target / (m - 1) for 2 windows;
		// without one (Reflex) a probe one step down after 2 windows on the cap (the check above reverts it).
		if (m > 2 && !on_cap) free_n = 0;
		if (m > 2 && on_cap && nodown == 0) {
			const bool known = free_real > 0.0;
			// Measured only: free >= 1.1 target / (m - 1) (FSR3's free rate, or a recent rate at m - 1); else stay.
			const double need = 1.1 * T / (m - 1);
			const bool   lower = free[m - 1] > 0.0 && age[m - 1] < 30 && free[m - 1] >= need;
			free_n             = (known && free_real >= need) || (!known && lower) ? free_n + 1 : 0;
			if (free_n >= 2) {
				noup      = 10;
				down_from = m;
				free_n    = 0;
				return go(m - 1, "on the cap; the lower level's free rate reaches the target");
			}
		}
		const int up = m + 1;
		if (up <= maxm && block[up] == 0 && real * m < 0.97 * T && real >= MinReal(up)) {
			auto take = [&] {
				check_from  = m;
				check_real  = real;
				check_shown = std::min(real * m, T);
				return go(up, "the next level shows more (estimate)");
			};
			// The real rate is the game's own cap (56-60): the next level within target * 1.03 keeps it, so no estimate.
			// Under 90% of the target for 5 s with real meeting the next level's minimum: probe it (the check reverts).
			if (noup == 0 && (real * up <= 1.03 * T || short_n >= 5)) return take();
			// Otherwise the estimate: meets the minimum and shows >= 5% more, or reaches the target with real capped by
			// at most ~10% (60 -> 56 for x4 = 224).
			const double est   = Estimate(up, real, m);
			const bool   reach = real * up >= T && T / up >= 0.9 * real;
			if (noup == 0 && est >= MinReal(up) && (reach || std::min(est * up, T) >= 1.05 * std::min(real * m, T))) {
				if (++up_n >= 2) return take();
				return m;
			}
		}
		up_n = 0;
		return m;
	}

	// 10-29 (ds-dlssg-pace-1029), DLSS-G only (FSR3 keeps Window): "real first". The real rate is never held under
	// what the game and the GPU make just to fill a multiplier. Logs 20261029-201439 / 201911 / 211510: Window climbed
	// x3 -> x6 while each step cost real frames (201911: 43.9 -> 41.6 -> 39.4 -> 37.1 real) because the screen showed
	// < (q - 0.05) x target, and at x6 the real rate sat on the pacer's cap target / 6 = 37.3 (render idle 4 ms, the
	// rest of the 26.8 ms frame held by the limiter); the screen then moved 1.3-1.4 ms off the game's own step on
	// average (13-18% of the frames > 2 ms) against FSR3 x4's 0.74 ms (3%) at 38-42 real, under its cap. The quality
	// rule's probe never ran (real under the cap most windows, and x5's old rate "short").
	// Rules (cap(m) = T / m, the pacer's interval that keeps the shown rate under the VRR ceiling):
	// - near the cap (real >= cap / Headroom(), 1.05: 95%) for 2 windows: the limiter, not the GPU, sets the real rate:
	//   one generated frame less (cap(m - 1) lets the game run free).
	// - up one level only when the real rate fits under the next cap with the same margin (real <= cap(m + 1) /
	//   Headroom()) for 3 windows, and meets that level's minimum; so the largest m whose cap stays above the real rate:
	//   the screen as close to the target as the GPU allows without a throttled real frame (the step's own cost only
	//   lowers the real rate, so it does not come back down: no ping-pong). The game held at its 60 stays x3 on 224
	//   (180 shown): x4 would hold it at 56. KYTY_DYN_HEADROOM=<1.0-1.5> sets the margin.
	// - 2 windows after a step up: when it cost more than MaxLoss() (12%) of the real rate (more than a generated
	//   frame's GPU cost: the presentation, not the GPU), back down and that level blocked 30 s (doubling, up to 4 min).
	// - under the level's minimum (MinReal) for 2 windows: down. Hitches / loading (real < 60% of the window before)
	//   and windows DLSS-G showed no generated frame (skip) decide nothing.
	static double Headroom() {
		static const double v = [] {
			const char* e = std::getenv("KYTY_DYN_HEADROOM");
			return e != nullptr && *e != '\0' ? std::clamp(std::atof(e), 1.0, 1.5) : 1.05;
		}();
		return v;
	}
	static double UpMargin() { return Headroom(); }
	static double MaxLoss() { return 0.12; }

	int  near_n = 0, fit_n = 0; // real-first: windows in a row near the cap / fitting under the next cap
	int WindowRealFirst(int m, double real, double T, int maxm, bool hold = false, bool skip = false) {
		if (hold) {
			*this = DynMultiplier {};
			return std::clamp(std::min(m, maxm), 2, kMax);
		}
		for (auto& b: block) b = std::max(b - 1, 0);
		for (auto& a: age) ++a;
		while (maxm > 2 && T / maxm < MinReal(maxm)) --maxm;
		maxm = std::clamp(maxm, 2, kMax);
		const double cap = T / m;
		next_est         = 0.0;
		if (skip) { // DLSS-G showed only the real frames: nothing to learn about m
			prev_real = 0.0;
			near_n = fit_n = low_n = 0;
			last_free = real;
			return std::min(m, maxm);
		}
		if (settle > 0) {
			--settle;
			prev_real = real;
			last_free = real;
			return m;
		}
		const double before = prev_real;
		prev_real           = real;
		if (before > 0.0 && real < 0.6 * before) {
			last_free = before;
			return m;
		}
		const bool at_cap = real >= cap / Headroom();
		if (!at_cap) free[m] = (free[m] > 0.0 && age[m] < 30) ? 0.5 * free[m] + 0.5 * real : real;
		age[m]    = 0;
		last_free = at_cap ? std::max(real, free[m]) : free[m];
		near_n    = at_cap ? near_n + 1 : 0;
		low_n     = real < MinReal(m) ? low_n + 1 : 0;
		const int    up  = m + 1;
		const bool   fit = up <= maxm && real <= T / up / UpMargin() && real >= MinReal(up);
		fit_n            = fit ? fit_n + 1 : 0;
		auto go = [&](int to, const char* why) {
			reason   = why;
			next_est = to < m ? std::max(real, free[to]) : real;
			settle   = 2;
			near_n = fit_n = low_n = 0;
			return to;
		};
		if (check_from != 0) {
			const int from = check_from;
			check_from     = 0;
			if (check_real > 0.0 && real > 0.0) {
				const double c = (1.0 / real - 1.0 / check_real) / std::max(m - from, 1);
				if (c > 0.0) cost = cost > 0.0 ? 0.5 * cost + 0.5 * c : c;
			}
			if (real < (1.0 - MaxLoss()) * check_real || real < MinReal(m)) {
				reverts[m] = std::min(reverts[m] + 1, 4);
				block[m]   = 30 << (reverts[m] - 1);
				free[m]    = 0.0;
				std::printf("[dyn] x%d cost the real rate %.1f -> %.1f fps (%.0f%%): back to x%d, x%d blocked %d s\n", m, check_real,
				            real, 100.0 * (1.0 - real / check_real), from, m, block[m]);
				return go(from, "the step up cost more than a generated frame's GPU time (real rate fell)");
			}
			reverts[m] = 0;
		}
		if (m > maxm) return go(maxm, "over the largest level allowed now");
		if (m > 2 && near_n >= 2) return go(m - 1, "real on the cap target / m for 2 s (held by the limiter): one generated frame less");
		if (m > 2 && low_n >= 2) return go(m - 1, "real under the level's minimum for 2 s");
		if (fit_n >= 3 && block[up] == 0) {
			check_from = m;
			check_real = real;
			return go(up, "real fits under the next level's cap for 3 s (no real frame throttled)");
		}
		return m;
	}
};

} // namespace Kyty::Local

#endif
