#pragma once
// (copia1) KYTY_COPIA1=1 (off by default; needs KYTY_DS_LINEAR_COPY_HOOK=1): the game's linear copies announced by
// consecutive sealed hook packets (local/ds-linear-copy-hook.h: announcement + the game's own packets of the copy,
// then the next announcement, with nothing of the game's between them but one-dword NOPs) are made together: one
// check of DsLinearCopy's terms, one PopPendingOperations, one render lock and one census scope for the run, the
// copies themselves one by one in the game's order (each still through TryLinearCopyRanges: CPU copy of the pages
// that differ, or a GPU copy in the buffer cache's copy run). Nothing is reordered: a copy that reads what an earlier
// one wrote sees it, as before. The run ends (a "flush", by reason) at any other packet, at an announcement that is not
// sealed, at the end of the command buffer, at a copy not made (its game's packets run before any later copy) and at
// MaxRun copies. A run of one is the old path.
//
// "[copia1]" line every 5 s (render thread): copies a frame, runs (the [render] line's linear_copy "dispatches") a
// frame, ms a frame inside the runs, and the flush reasons a frame.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Copia1 {

constexpr uint32_t MaxRun = 64;

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_COPIA1");
		return text != nullptr && text[0] == '1' && text[1] == '\0';
	}();
	return on;
}

enum Flush : uint32_t { OtherPacket, NotSealed, BufferEnd, Refused, Full, Flushes };
inline const char* const FlushNames[Flushes] {"other_packet", "not_sealed", "buffer_end", "refused", "full"};

struct Stats {
	uint64_t copies = 0, runs = 0, ns = 0;
	uint64_t flushes[Flushes] {};
	std::chrono::steady_clock::time_point start {};
};
inline Stats g_stats; // (render thread only)

// A frame passed (the [render] line's frame count is not reachable here: flips counted by the caller of NoteFrame).
inline std::atomic<uint64_t> g_frames {0};
inline void NoteFrame() { g_frames.fetch_add(1, std::memory_order_relaxed); }

inline void NoteRun(uint32_t copies, uint64_t ns, Flush why) {
	auto& s = g_stats;
	s.copies += copies;
	++s.runs;
	s.ns += ns;
	++s.flushes[why];
	const auto now = std::chrono::steady_clock::now();
	if (s.start == std::chrono::steady_clock::time_point {}) {
		s.start = now;
		g_frames.store(0, std::memory_order_relaxed);
		return;
	}
	if (now - s.start < std::chrono::seconds(5)) return;
	const double frames = static_cast<double>(std::max<uint64_t>(1, g_frames.exchange(0, std::memory_order_relaxed)));
	std::printf("[copia1] %.1f s, %.0f frames: copies/frame=%.1f runs/frame=%.1f (%.2f copies/run) ms/frame=%.3f | flush",
	            std::chrono::duration<double>(now - s.start).count(), frames, s.copies / frames, s.runs / frames,
	            s.runs ? static_cast<double>(s.copies) / s.runs : 0.0, s.ns / 1e6 / frames);
	for (uint32_t i = 0; i < Flushes; ++i) std::printf(" %s=%.1f", FlushNames[i], s.flushes[i] / frames);
	std::printf("\n");
	std::fflush(stdout);
	s = Stats {};
	s.start = now;
}

} // namespace Copia1
