#ifndef EMULATOR_SRC_LOCAL_TRAD1_H_
#define EMULATOR_SRC_LOCAL_TRAD1_H_
// (trad1) KYTY_TRAD1=1|2 (off by default; read once). The table path's written V#s (TableBlocks, table-xpr.inc): a range
// a shader writes is obtained as written (BufferCache::ObtainBuffer) at every use; the memo there (TableXpr::WrittenRange)
// skips the second ObtainBuffer only inside one tick and while the range's 4 MiB tracker region saw no CPU write and no
// readback/feedback moved WrittenObtainEpoch anywhere. In a frame ~4.8 K written V#s, ~1.9 K went through the full
// ObtainBuffer (log 20261118-195619 "blocks written=4839 obtained=1856"): every submission slice (~80 a frame, and
// KYTY_DISPATCH_BATCH's flushes) starts a new tick.
//   1: a written V# whose last full obtain (this proof's) still holds on the terms ObtainBuffer's own write memo
//      (m_write_memo) stands on - no buffer registered or unregistered since (registration epoch), the same buffer live,
//      no GPU-written byte given back since (BufferCache's coverage epoch: its pages still GPU-owned and the range still
//      in the GPU-modified set, so SynchronizeBuffer would upload nothing) - only renotes the write for this tick:
//      BufferCache::Trad1RenoteWritten (policy, readback drain first, LRU touch, copy feedback invalidated, GPU write and
//      native write noted: what ObtainBuffer's memo path does) and takes the device address it had. No page-table lookup,
//      no SynchronizeBuffer, no range Add under the mutex, no FindBuffer / epochs after it.
//   2: verification: where 1 would take the shortcut, the old path runs instead (its result is the one used) and is
//      compared: the buffer FindBuffer gives, the device address, and whether SynchronizeBuffer would have uploaded
//      (pages not all GPU-owned). Each mismatch is a DIFERENCIA (the first 16 printed).
// A "[trad1]" line every 5 s: hits, the misses by cause, us per call of each path (sampled), the estimated ms a frame
// saved, DIFERENCIAS.
// trad1b: the shortcut's entries are keyed by the range (clamped address, size), not by the V#'s words / entry, and kept
// across the special generation's moves; the coverage proof is per 4 MiB tracker region (BufferCache::Trad1CoverageMoved)
// instead of the global coverage epoch. The line's frames are LiveControl::g_flips (TextureCache::CurrentFrame stayed 0).
// Off: one load of a static per written V# that misses the old memo, and per V# proof made again.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

namespace Trad1 {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_TRAD1");
		const auto  m    = text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
		if (m != 0) {
			std::printf("[trad1] on (KYTY_TRAD1=%u): written V#s renoted per tick without a second ObtainBuffer%s; "
			            "a [trad1] line every 5 s\n",
			            m, m == 2 ? " - VERIFY: the old path runs and is compared (DIFERENCIAS)" : "");
			std::fflush(stdout);
		}
		return m > 2 ? 1u : m;
	}();
	return mode;
}

// Why a written V# that missed the old memo did not take the shortcut.
enum Miss : uint32_t {
	NoProof,      // no earlier full obtain of these words kept (new V#, formatted, speculation, another executor...)
	Registration, // a buffer was registered or unregistered since
	BufferGone,   // its buffer was deleted or replaced
	Coverage,     // GPU-written bytes were given back since (readback, download, feedback, discard)
	Refused,      // Trad1RenoteWritten refused it (after the readback drain)
	MissCount
};

struct Stats {
	uint64_t hits = 0;                     // shortcut taken (mode 1) / it would have been (mode 2)
	uint64_t old_memo = 0;                 // the old in-tick memo's hits (unchanged)
	std::array<uint64_t, MissCount> miss {}; // full ObtainBuffer, by cause
	uint64_t checked = 0, differences = 0, dirty = 0; // mode 2
	// Sampled timings (rdtsc cycles), 1 in 8 of each path.
	uint64_t full_cycles = 0, full_samples = 0, fast_cycles = 0, fast_samples = 0;
	uint64_t full_seen = 0, fast_seen = 0;
	// (trad1b) The V#-keyed proofs (TableXpr::WrittenRange) made again: other words in the entry (a new V# or another
	// one sharing it), or the special generation moved (mappings, image starts).
	uint64_t reset_words = 0, reset_special = 0;
};
inline Stats g;              // the render thread's (TableBlocks' main executor)
inline Stats g_last;
inline int64_t  g_last_ns    = 0;
inline uint64_t g_last_tsc   = 0;
inline uint64_t g_last_frame = 0;
inline uint64_t g_calls      = 0;
// Cumulated over the run, for the summary.
inline double   g_total_saved_ms = 0;
inline uint64_t g_total_frames   = 0;

[[nodiscard]] inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Sampling: true for 1 in 8 calls of a path.
[[nodiscard]] inline bool SampleFull() { return (g.full_seen++ & 7u) == 0; }
[[nodiscard]] inline bool SampleFast() { return (g.fast_seen++ & 7u) == 0; }

inline void NoteDifference(const char* what, uint64_t address, uint64_t size, uint64_t a, uint64_t b) {
	++g.differences;
	if (g.differences <= 16) {
		std::printf("[trad1] DIFERENCIA %s at 0x%llx+0x%llx: shortcut 0x%llx, old path 0x%llx\n", what,
		            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
		            static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
		std::fflush(stdout);
	}
}

// Every 512 written V#s: the clock; every 5 s a line. `frame`: TextureCache::CurrentFrame.
inline void MaybeReport(uint64_t frame) {
	if ((++g_calls & 511u) != 0) return;
	const auto now = NowNs();
	const auto tsc = __rdtsc();
	if (g_last_ns == 0) {
		g_last_ns = now, g_last_tsc = tsc, g_last_frame = frame, g_last = g;
		return;
	}
	if (now - g_last_ns < 5'000'000'000) return;
	const double seconds = static_cast<double>(now - g_last_ns) / 1e9;
	const double ns_per_cycle =
	    tsc > g_last_tsc ? static_cast<double>(now - g_last_ns) / static_cast<double>(tsc - g_last_tsc) : 0.0;
	const double frames = frame > g_last_frame ? static_cast<double>(frame - g_last_frame) : 0.0;
	const auto   d      = [](uint64_t a, uint64_t b) { return static_cast<double>(a - b); };
	const double per    = frames > 0 ? frames : seconds; // a frame, else a second
	const char*  unit   = frames > 0 ? "frame" : "s";
	const double full_us = g.full_samples > g_last.full_samples
	                           ? d(g.full_cycles, g_last.full_cycles) / d(g.full_samples, g_last.full_samples) * ns_per_cycle / 1e3
	                           : 0.0;
	const double fast_us = g.fast_samples > g_last.fast_samples
	                           ? d(g.fast_cycles, g_last.fast_cycles) / d(g.fast_samples, g_last.fast_samples) * ns_per_cycle / 1e3
	                           : 0.0;
	const double hits = d(g.hits, g_last.hits);
	double       misses = 0;
	for (uint32_t i = 0; i < MissCount; ++i) misses += d(g.miss[i], g_last.miss[i]);
	// Saved: each shortcut instead of the full path (mode 2: what mode 1 would save; the shortcut's own cost there is
	// not measured, the full path's is taken as all of it minus mode 1's typical shortcut, 0 when unknown).
	const double saved_ms = full_us > 0 && hits > 0 ? hits * (full_us - fast_us) / 1e3 : 0.0;
	std::printf("[trad1] modo=%u %.1f s, %.0f frames: aciertos=%.1f/%s fallos=%.1f/%s (sin_prueba %.1f, registro %.1f, "
	            "buffer %.1f, cobertura %.1f, rechazo %.1f) memo_viejo=%.1f/%s pruebas_V#_rehechas palabras=%.1f "
	            "especial=%.1f | us/llamada: via_vieja %.3f atajo %.3f | "
	            "%s %.3f ms/%s | DIFERENCIAS %llu (comprobados %.1f/%s, sucios %llu)\n",
	            Mode(), seconds, frames, hits / per, unit, misses / per, unit, d(g.miss[NoProof], g_last.miss[NoProof]) / per,
	            d(g.miss[Registration], g_last.miss[Registration]) / per, d(g.miss[BufferGone], g_last.miss[BufferGone]) / per,
	            d(g.miss[Coverage], g_last.miss[Coverage]) / per, d(g.miss[Refused], g_last.miss[Refused]) / per,
	            d(g.old_memo, g_last.old_memo) / per, unit, d(g.reset_words, g_last.reset_words) / per,
	            d(g.reset_special, g_last.reset_special) / per, full_us, fast_us,
	            Mode() == 2 ? "techo (=2: via vieja de los elegibles, atajo sin medir)" : "ahorro estimado", saved_ms / per, unit,
	            static_cast<unsigned long long>(g.differences), d(g.checked, g_last.checked) / per, unit,
	            static_cast<unsigned long long>(g.dirty));
	if (frames > 0) {
		g_total_saved_ms += saved_ms;
		g_total_frames += static_cast<uint64_t>(frames);
		std::printf("[trad1] acumulado: %llu frames, ahorro estimado medio %.3f ms/frame, DIFERENCIAS %llu\n",
		            static_cast<unsigned long long>(g_total_frames),
		            g_total_frames != 0 ? g_total_saved_ms / static_cast<double>(g_total_frames) : 0.0,
		            static_cast<unsigned long long>(g.differences));
	}
	std::fflush(stdout);
	g_last_ns = now, g_last_tsc = tsc, g_last_frame = frame, g_last = g;
}

} // namespace Trad1

#endif // EMULATOR_SRC_LOCAL_TRAD1_H_
