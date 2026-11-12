#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SPECULATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SPECULATION_H_

#include <array>
#include <cstdint>

// Speculation (gt7-spec-1112; KYTY_SPECULATE=<n>, 0 or unset: off).
//
// Demon's Souls' 40 fps (des-black-1111, chenxiao07 experiment/perf-40fps-20260926) came from
// translating the command stream ahead of the GPU thread on worker threads: a scanner follows the
// register writes on a command processor of its own, workers prepare each draw's expensive work
// from that speculated state, the results are published in packet order, and the GPU thread only
// validates them (same packet, same inputs, the guest bytes read still the same and clean) and
// emits; on any difference it does the work itself as before.
//
// Here that pipeline is the draw-prep scanner (drawPrep.h: Thread_DrawPrep follows the registers,
// Thread_DrawPrepWalk walks the descriptors, ProgramCache::TakeScanned validates). KYTY_SPECULATE
// builds on it:
//  - n walk workers at least (KYTY_DRAW_PREP_WORKERS still sets more), so the scanner stays ahead
//    of the GPU thread;
//  - ProgramCache::Get takes a validated speculated refresh on a short path (SpecFastPath) that
//    skips the walk's set-up (runtime, page cache, profiler zone, report checks); every 64th one
//    goes the classic way instead, to measure what the short path saves.
// The counters are logged every 10 s on a "Spec" line after the "GPU phases" line.
namespace Libs::Graphics::Speculation {

// KYTY_SPECULATE's value (0: off).
[[nodiscard]] uint32_t Threads();
[[nodiscard]] inline bool Enabled() {
	return Threads() != 0;
}

enum class Counter : uint32_t {
	FastHit,       // program gets served by the short path
	FastMiss,      // the short path found nothing valid: the classic path ran
	Calibration,   // gets sent the classic way to measure it
	Count,
};

// GPU thread only (no atomics needed).
void Add(Counter counter, uint64_t value = 1);
// GPU thread: the cycles of a short-path hit, and of a classic scanned get (calibration).
void AddFastCycles(uint64_t cycles);
void AddClassicCycles(uint64_t cycles);
// GPU thread: whether this hit should go the classic way (every 64th).
[[nodiscard]] bool CalibrateNext();

// GPU thread, after the "GPU phases" line: the "Spec" line (Log::WriteUnmuted).
void Report(double cycles_per_us, double seconds);

} // namespace Libs::Graphics::Speculation

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SPECULATION_H_
