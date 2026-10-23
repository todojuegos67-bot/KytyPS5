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
		auto go = [&](int to) {
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
				block[m] = 10;
				free[m]  = 0.0;
				return go(from);
			}
		}
		// Down after 2 windows in a row below the level's minimum (to the largest lower level whose known rate meets its own).
		if (m > 2 && (low_n >= 2 || m > maxm)) {
			int to = std::min(m - 1, maxm);
			while (to > 2 && free[to] > 0.0 && age[to] < 30 && free[to] < MinReal(to)) --to;
			return go(to);
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
				return go(from);
			}
			nodown_len = 15;
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
					return go(m - 1);
				}
				if (!fits && pinned >= 3) { // held on target / m: the game may run faster with fewer generated frames
					std::printf("[dyn] headroom probe: x%d -> x%d (real %.1f held at the cap %.1f)\n", m, m - 1, real, cap);
					noup      = 5;
					down_from = m;
					return go(m - 1);
				}
			} else {
				qdown_n = 0;
			}
			const int up = m + 1;
			if (up <= maxm && block[up] == 0 && noup == 0 && qup_n >= 3 && real >= MinReal(up)) {
				std::printf("[dyn] x%d shows %.1f < %.1f for 3 s: up to x%d\n", m, shown, (q - 0.05) * T, up);
				check_from  = m;
				check_real  = real;
				check_shown = shown;
				return go(up);
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
				return go(m - 1);
			}
		}
		const int up = m + 1;
		if (up <= maxm && block[up] == 0 && real * m < 0.97 * T && real >= MinReal(up)) {
			auto take = [&] {
				check_from  = m;
				check_real  = real;
				check_shown = std::min(real * m, T);
				return go(up);
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
};

} // namespace Kyty::Local

#endif
