#pragma once
// (render5) Small savings of the render thread's own time, each behind its own switch, off by default. Each switch is
// read once, at its first use (the port sets its variables after the process started), and printed once:
//   KYTY_CENSUS_LIGHT (live-census.h): the per-draw and per-dispatch census scopes (two TSC reads and a few sums each,
//     ~40 000 a frame at Boletaria) are taken only while something reads them (KYTY_RENDER_STATS, the live census).
//   KYTY_REG_LANE (graphicsRun.cpp, CommandProcessor::ProcessPm4): register and index state packets that follow each
//     other run in a tight loop, without the per-packet work of the general walk (the guest commands and the copy
//     feedback between packets, the chunk, speculation and hook checks).
//   KYTY_DRAW_FLAGS (native-xpr.inc, table-xpr.inc): the process-constant answers a table draw asks the frame generation
//     and RT code for at every draw (FrameGen::SsrWanted, ContactShadowsWanted, RtSunWanted, RtParticlesDebug and the
//     constant half of Capturing) are read once.
// 1: on. 2: verification: the old path runs (what is drawn is the same as with 0) and the new one is checked against
// it: the differences are counted and printed in a [render5] line every 5 s (any KYTY_RENDER_STATS) and at the end of
// the [render] line. 0 or unset: as before.

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Render5 {

enum Switch : uint32_t { CensusLight, RegLane, DrawFlags, Switches };
inline constexpr const char* SwitchNames[Switches] = {"KYTY_CENSUS_LIGHT", "KYTY_REG_LANE", "KYTY_DRAW_FLAGS"};
inline std::atomic<int32_t> g_mode[Switches] {-1, -1, -1};

// The switch's mode, read at the first call (any thread; the first one to read it prints it).
[[nodiscard]] inline uint32_t ReadMode(Switch which) {
	const char* text = std::getenv(SwitchNames[which]);
	int32_t     mode = 0;
	if (text != nullptr && (text[0] == '1' || text[0] == '2') && text[1] == '\0') mode = text[0] - '0';
	int32_t expected = -1;
	if (g_mode[which].compare_exchange_strong(expected, mode, std::memory_order_relaxed)) {
		static constexpr const char* what[Switches] = {
		    "the per-draw/per-dispatch census scopes only while KYTY_RENDER_STATS or the live census reads them",
		    "register and index state packets in a tight loop (ProcessPm4)",
		    "the table draws' process-constant frame generation / RT answers read once"};
		std::printf("render5: %s=%d: %s%s\n", SwitchNames[which], mode, what[which],
		            mode == 0 ? " - off (as before)" : mode == 1 ? " - on" : " - verification (old path runs; differences counted in [render5])");
		std::fflush(stdout);
		return static_cast<uint32_t>(mode);
	}
	return static_cast<uint32_t>(expected);
}
[[nodiscard]] inline uint32_t Mode(Switch which) {
	const int32_t mode = g_mode[which].load(std::memory_order_relaxed);
	return mode >= 0 ? static_cast<uint32_t>(mode) : ReadMode(which);
}

// Counters. Lane and census ones: the render thread's only (plain sums). Draw flag ones: any executor's thread
// (relaxed atomics, taken in verification only).
enum LaneCounter : uint32_t {
	LaneRuns,       // runs the lane took (mode 1) or predicted (mode 2)
	LanePackets,    // their packets
	LaneSame,       // mode 2: runs the old path walked exactly as predicted (same buffer, packets and end)
	LaneDifferent,  // mode 2: runs it walked otherwise (a mismatch: the lane would have parsed differently)
	LanePostponed,  // mode 2: guest commands the old path ran inside a predicted run (mode 1 runs them after it)
	LaneFeedback,   // mode 2: copy feedback written back inside a predicted run (mode 1: after it)
	LaneCounters
};
inline uint64_t g_lane[LaneCounters] {};
enum CensusCounter : uint32_t {
	CensusSkipped,     // mode 1: scopes not taken
	CensusWouldSkip,   // mode 2: scopes mode 1 would not take
	CensusCounters
};
inline uint64_t g_census[CensusCounters] {};
inline uint64_t g_census_scope_cycles = 0; // mode 2: a scope's cost measured at start (TSC cycles)
enum FlagCounter : uint32_t { FlagChecks, FlagMismatches, FlagCounters };
inline std::atomic<uint64_t> g_flags[FlagCounters] {};

} // namespace Render5
