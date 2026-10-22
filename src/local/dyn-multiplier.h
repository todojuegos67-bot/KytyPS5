#ifndef KYTY_LOCAL_DYN_MULTIPLIER_H_
#define KYTY_LOCAL_DYN_MULTIPLIER_H_

// 10-22: the dynamic frame-generation multiplier shared by DLSS-G (frame-gen.cpp) and FSR3 MFG (fsr3-fg.cpp).
// Decided on the GPU's free render rate, not the rate seen under the current multiplier (generating more frames
// costs GPU time; "largest m with real * m <= target" read on that lowered rate spiralled x4 -> x6 at 9 real fps):
// - free[m]: the real rate last measured at multiplier m (EMA, not sitting on the target / m cap); the rate at m + 1
//   is estimated as 1 / (1 / free[m] + cost), cost = measured seconds per extra generated frame (from switches),
//   8% of the frame until one was measured.
// - Minimum real fps per multiplier: x2 30, x3 40, x4 45, x5/x6 50; never up when the estimate for m + 1 is below its
//   minimum, down at once when real falls below the current one's; m only where target / m >= its minimum.
// - Up one level at a time, only when it shows >= 5% more (min(real * m, target)); 2 windows after a step up, the
//   shown rate is checked against the one before: not >= 5% higher (or real under the minimum) -> back down and that
//   level blocked for 10 s. The level that reaches the target with real capped by <= 10% is preferred (60 real on
//   224: x4 at 56 real, kept when it shows >= 97% of the target).
// - The cap (real <= target / m) binds only when real * m would pass the target; when real sits on it for 2 windows
//   and the lower multiplier's free rate also reaches the target, one step down (no step up for 10 s).
#include <algorithm>
#include <array>
#include <cmath>

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

	static double MinReal(int m) { return m <= 2 ? 30.0 : m == 3 ? 40.0 : m == 4 ? 45.0 : 50.0; }

	double Estimate(int m, double base_real, int base_m) const {
		if (free[m] > 0.0 && age[m] < 10) return free[m]; // (recent only: the scene moves on)
		const double c = cost > 0.0 ? cost : 0.08 / std::max(base_real, 1.0);
		return 1.0 / (1.0 / std::max(base_real, 1.0) + c * (m - base_m));
	}

	// Once a second. m: current multiplier (>= 2), real: this window's real fps, T: shown target, maxm: the largest
	// multiplier allowed now (cinematics pass 3). Returns the multiplier for the next window.
	// hold: a cinematic, held cutscene or movie (real frames held to 60/30): nothing learned or blocked, all state reset
	// so the levels and estimates start fresh when it ends.
	int Window(int m, double real, double T, int maxm, bool hold = false) {
		if (hold) {
			*this = DynMultiplier {};
			return std::clamp(std::min(m, maxm), 2, kMax);
		}
		for (auto& b: block) b = std::max(b - 1, 0);
		for (auto& a: age) ++a;
		if (noup > 0) --noup;
		while (maxm > 2 && T / maxm < MinReal(maxm)) --maxm;
		maxm = std::clamp(maxm, 2, kMax);
		const double cap = T / m;
		const bool   on_cap = real >= 0.98 * cap;
		pinned = on_cap ? pinned + 1 : 0;
		if (settle > 0) {
			--settle;
			last_free = Estimate(m, real, m);
			return m;
		}
		// Measure: an EMA of the free rate (a window on the cap is only a lower bound).
		if (!on_cap || free[m] < real) free[m] = (free[m] > 0.0 && age[m] < 30 && !on_cap) ? 0.5 * free[m] + 0.5 * real : real;
		age[m]    = 0;
		// A higher level's stored rate far under what this level measures now was learned in another scene: dropped.
		for (int k = m + 1; k <= kMax; ++k)
			if (free[k] > 0.0 && free[k] < 0.75 * real) free[k] = 0.0;
		last_free = free[m];
		auto go = [&](int to) {
			settle = 1;
			up_n   = 0;
			pinned = 0;
			return to;
		};
		// A step up under test: keep it only with >= 5% more shown and real at the level's minimum.
		if (check_from != 0) {
			const int from = check_from;
			check_from     = 0;
			const double shown = std::min(real * m, T);
			if (real > 0.0 && check_real > 0.0) {
				const double c = (1.0 / real - 1.0 / check_real) / std::max(m - from, 1);
				if (c > 0.0) cost = cost > 0.0 ? 0.5 * cost + 0.5 * c : c;
			}
			if ((shown < 1.05 * check_shown && shown < 0.97 * T) || real < MinReal(m)) {
				block[m] = 10;
				return go(from);
			}
		}
		// Down at once below the level's minimum (to the largest lower level whose known rate meets its own).
		if (m > 2 && (real < MinReal(m) || m > maxm)) {
			int to = std::min(m - 1, maxm);
			while (to > 2 && free[to] > 0.0 && age[to] < 30 && free[to] < MinReal(to)) --to;
			return go(to);
		}
		// On the cap: the lower multiplier also reaches the target -> fewer generated frames for the same shown rate.
		if (pinned >= 2 && m > 2 && free[m - 1] > 0.0 && age[m - 1] < 30 && free[m - 1] * (m - 1) >= 0.98 * T) {
			noup = 10;
			return go(m - 1);
		}
		// Up one level: room under the target, the estimate meets the minimum and shows >= 5% more.
		const int up = m + 1;
		if (up <= maxm && noup == 0 && block[up] == 0 && real * m < 0.97 * T) {
			const double est = Estimate(up, real, m);
			// Reaching the target with real capped by at most ~10% (60 -> 56 for x4 = 224) is preferred.
			const bool reach = real * up >= T && T / up >= 0.9 * real;
			if (est >= MinReal(up) && (reach || std::min(est * up, T) >= 1.05 * std::min(real * m, T))) {
				if (++up_n >= 2) {
					check_from  = m;
					check_real  = real;
					check_shown = std::min(real * m, T);
					return go(up);
				}
				return m;
			}
		}
		up_n = 0;
		return m;
	}
};

} // namespace Kyty::Local

#endif
