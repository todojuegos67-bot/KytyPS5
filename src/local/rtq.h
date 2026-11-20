#ifndef EMULATOR_SRC_LOCAL_RTQ_H_
#define EMULATOR_SRC_LOCAL_RTQ_H_
// (rtq1) KYTY_RTQ=1: the improved versions of the ray-traced effects (default off in this build; the launchers turn it
// on). Today only "Reflejos RT" has one (rt-reflections.cpp / .comp: the ray budget by contribution, a thread per 2x2
// trace block, the composite's early out where the trace marked nothing and their checks); "Luz volumetrica" and
// "Sombras sol (RT real)" are unchanged (see rtq1's INFORME.md: no cost data to act on yet).
// KYTY_RTQ_AB=1 (implies KYTY_RTQ): 5 s windows alternating OLD (even windows) and NEW (odd), the paired line
// "[rtq] reflejos A/B" every 10 s. F10 (KYTY_RTQ only): the debug view on / off, and a notice OLD / NEW at each window
// while it is on (screenshots).

#include <atomic>
#include <chrono>
#include <cstdlib>

namespace Rtq {

inline bool Ab() {
	static const bool v = [] {
		const char* a = std::getenv("KYTY_RTQ_AB");
		return a != nullptr && a[0] == '1';
	}();
	return v;
}
inline bool Wanted() {
	static const bool v = [] {
		const char* e = std::getenv("KYTY_RTQ");
		return (e != nullptr && e[0] == '1') || Ab();
	}();
	return v;
}
// The 5 s window index since the first call.
inline long long WindowIndex() {
	static const auto t0 = std::chrono::steady_clock::now();
	return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count() / 5;
}
// The new versions this frame: KYTY_RTQ alone always; with KYTY_RTQ_AB the odd windows.
inline bool NewPhase() {
	if (!Wanted()) return false;
	if (!Ab()) return true;
	return (WindowIndex() & 1) != 0;
}
inline std::atomic<bool> debug_view {false}; // F10 (window.cpp)
inline bool DebugView() { return Wanted() && debug_view.load(std::memory_order_relaxed); }

} // namespace Rtq

#endif // EMULATOR_SRC_LOCAL_RTQ_H_
