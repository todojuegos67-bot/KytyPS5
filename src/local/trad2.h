#ifndef EMULATOR_SRC_LOCAL_TRAD2_H_
#define EMULATOR_SRC_LOCAL_TRAD2_H_
// (trad2) KYTY_TRAD2=1|2 (off by default; read once). The BDA region requests of the dispatches and draws that read
// guest memory by address (GpuResourceManager::PrepareBdaReadRanges -> SynchronizeDirtyBdaRegions ->
// BufferCache::SynchronizeRegionRequest). With KYTY_BDA_DIRTY_REGIONS=1 only the 4 MiB tracker regions whose bit is set
// are visited, and a visited request whose region's CPU epoch moved is synchronized again: SynchronizeBuffersInRange
// (readback drain, the sync index, per piece the epoch, its stamp and SynchronizeBuffer's own dirty check). The region's
// epoch moves when ANY page of its 4 MiB becomes CPU-dirty (a guest write fault, a CP copy the emulator made through the
// backing view, the volatile re-dirty of each submission, a no-watch page announced): every request of the region is
// synchronized again, also the ones none of whose pages is dirty. In a frame at Boletaria 1-1 ~500-830 such syncs,
// region_skips 0, 0.46-0.69 ms of the render thread (log 20261118-195619, [render]: region_syncs, region_sync_us), about
// one per BDA call (bda_range_calls ~730).
//   1: a request none of whose pages is CPU-dirty (MemoryTracker::IsRegionCpuModified over the request: what each piece's
//      SynchronizeBuffer checks first, and a piece lies inside its request) and with no no-watch page the policy could
//      give back (KYTY_DS_NEIGHBOR_SPLIT: PolicyBdaRead) is not walked: SynchronizeBuffersInRange would upload nothing
//      and change nothing (its stamps aside: they only let it skip that same check later). Its proof is made as the old
//      path makes it (the epochs read before the check, unchanged after it), so the next visit skips it as before. A
//      dirty request runs the old path (real uploads). 1 in 32 shortcuts runs the old path anyway (timed, and checked:
//      an upload made there is a DIFERENCIA).
//   2: verification: every would-be shortcut runs the old path instead (the one used) and is checked the same way.
// A "[trad2]" line every 5 s, per frame (LiveControl::g_flips): shortcuts, old-path syncs by result (dirty: real
// uploads, no-watch) and by why the old proof failed, us per call (sampled), estimated ms saved, DIFERENCIAS.
// Off: one load of a static per request that the old proof did not skip.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

namespace Trad2 {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_TRAD2");
		const auto  m    = text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
		if (m != 0) {
			std::printf("[trad2] on (KYTY_TRAD2=%u): BDA region requests with no CPU-dirty page are not walked again%s; "
			            "a [trad2] line every 5 s\n",
			            m, m == 2 ? " - VERIFY: the old path runs and is checked (DIFERENCIAS)" : "");
			std::fflush(stdout);
		}
		return m > 2 ? 1u : m;
	}();
	return mode;
}

// Why the old proof (the request's epochs) did not skip the request.
enum Why : uint32_t {
	Unproven,     // never proven, or left unproven by its last sync (cpu_epoch 0)
	EpochZero,    // the region's epoch is 0 now (a deferred write protection queued, untracked)
	Registration, // a buffer of its region was registered or unregistered since
	EpochMoved,   // some page of its 4 MiB region became CPU-dirty since (or the emulator wrote dirty pages)
	WhyCount
};

// What became of a request the old proof did not skip.
enum Result : uint32_t {
	Shortcut, // no CPU-dirty page, no no-watch page: not walked (mode 1) / would not be (mode 2)
	Dirty,    // some page of it CPU-dirty: the old path (uploads)
	NoWatch,  // a no-watch page the policy may give back: the old path
	ResultCount
};

struct Stats {
	std::array<uint64_t, WhyCount>    why {};
	std::array<uint64_t, ResultCount> result {};
	uint64_t proven = 0;                               // shortcuts that left the request proven
	uint64_t checked = 0, differences = 0;             // old path run over a would-be shortcut, and found doing work
	// Sampled timings (rdtsc cycles): the old path over clean requests (1 in 32 shortcuts in mode 1, 1 in 4 in mode 2),
	// over dirty ones (1 in 8), and the shortcut's check (1 in 8).
	uint64_t clean_cycles = 0, clean_samples = 0;
	uint64_t dirty_cycles = 0, dirty_samples = 0, dirty_seen = 0;
	uint64_t fast_cycles = 0, fast_samples = 0, fast_seen = 0;
	uint64_t shortcut_seen = 0;
};
inline Stats g; // the render thread's (BDA preparation runs there only: a speculation's waits for its commit)
inline Stats g_last;
inline int64_t  g_last_ns    = 0;
inline uint64_t g_last_tsc   = 0;
inline uint64_t g_last_frame = 0;
inline uint64_t g_calls      = 0;
inline double   g_total_saved_ms = 0;
inline uint64_t g_total_frames   = 0;
// Runs of dirty pages BufferCache::UploadDirtyRanges copied on this thread (mode 1/2 verification: the old path over a
// request found clean must copy none).
inline thread_local uint64_t g_upload_runs = 0;

[[nodiscard]] inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Mode 1: 1 in 32 shortcuts runs the old path (timed and checked); mode 2: all run it, 1 in 4 timed.
[[nodiscard]] inline bool ShortcutRunsOld() { return Mode() == 2 || (g.shortcut_seen++ & 31u) == 0; }
[[nodiscard]] inline bool SampleCleanTiming() { return Mode() == 1 || (g.checked & 3u) == 0; }
[[nodiscard]] inline bool SampleDirty() { return (g.dirty_seen++ & 7u) == 0; }
[[nodiscard]] inline bool SampleFast() { return (g.fast_seen++ & 7u) == 0; }

inline void NoteDifference(uint64_t address, uint64_t size, uint64_t runs) {
	++g.differences;
	if (g.differences <= 16) {
		std::printf("[trad2] DIFERENCIA at 0x%llx+0x%llx: the request had no CPU-dirty page, the old path uploaded %llu runs\n",
		            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
		            static_cast<unsigned long long>(runs));
		std::fflush(stdout);
	}
}

// Every 256 requests: the clock; every 5 s a line. `frame`: LiveControl::g_flips.
inline void MaybeReport(uint64_t frame) {
	if ((++g_calls & 255u) != 0) return;
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
	const double per    = frames > 0 ? frames : seconds;
	const char*  unit   = frames > 0 ? "frame" : "s";
	const auto   us     = [&](uint64_t c1, uint64_t c0, uint64_t n1, uint64_t n0) {
        return n1 > n0 ? d(c1, c0) / d(n1, n0) * ns_per_cycle / 1e3 : 0.0;
	};
	const double clean_us = us(g.clean_cycles, g_last.clean_cycles, g.clean_samples, g_last.clean_samples);
	const double dirty_us = us(g.dirty_cycles, g_last.dirty_cycles, g.dirty_samples, g_last.dirty_samples);
	const double fast_us  = us(g.fast_cycles, g_last.fast_cycles, g.fast_samples, g_last.fast_samples);
	const double hits     = d(g.result[Shortcut], g_last.result[Shortcut]);
	const double dirty    = d(g.result[Dirty], g_last.result[Dirty]);
	// Saved: each shortcut instead of the old path over a clean request (mode 2: what mode 1 would save).
	const double saved_ms = clean_us > fast_us && hits > 0 ? hits * (clean_us - fast_us) / 1e3 : 0.0;
	std::printf("[trad2] modo=%u %.1f s, %.0f frames: atajos=%.1f/%s (probados %.1f) via_vieja: sucias=%.1f no_watch=%.1f | "
	            "por que se visito: sin_prueba %.1f, epoca_0 %.1f, registro %.1f, epoca_movida %.1f | "
	            "us/llamada: via_vieja_limpia %.3f atajo %.3f via_vieja_sucia %.3f (sucias %.3f ms/%s) | "
	            "%s %.3f ms/%s | DIFERENCIAS %llu (comprobados %.1f/%s)\n",
	            Mode(), seconds, frames, hits / per, unit, d(g.proven, g_last.proven) / per, dirty / per,
	            d(g.result[NoWatch], g_last.result[NoWatch]) / per, d(g.why[Unproven], g_last.why[Unproven]) / per,
	            d(g.why[EpochZero], g_last.why[EpochZero]) / per, d(g.why[Registration], g_last.why[Registration]) / per,
	            d(g.why[EpochMoved], g_last.why[EpochMoved]) / per, clean_us, fast_us, dirty_us, dirty * dirty_us / 1e3 / per,
	            unit, Mode() == 2 ? "techo (=2)" : "ahorro estimado", saved_ms / per, unit,
	            static_cast<unsigned long long>(g.differences), d(g.checked, g_last.checked) / per, unit);
	if (frames > 0) {
		g_total_saved_ms += saved_ms;
		g_total_frames += static_cast<uint64_t>(frames);
		std::printf("[trad2] acumulado: %llu frames, ahorro estimado medio %.3f ms/frame, DIFERENCIAS %llu\n",
		            static_cast<unsigned long long>(g_total_frames),
		            g_total_frames != 0 ? g_total_saved_ms / static_cast<double>(g_total_frames) : 0.0,
		            static_cast<unsigned long long>(g.differences));
	}
	std::fflush(stdout);
	g_last_ns = now, g_last_tsc = tsc, g_last_frame = frame, g_last = g;
}

} // namespace Trad2

#endif // EMULATOR_SRC_LOCAL_TRAD2_H_
