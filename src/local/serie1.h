#pragma once
// (serie1) KYTY_SERIE1: the XPR culling's command buffer runs on the GPU's second queue, so that the game's reads of its
// bits stop waiting behind the graphics work queued before it (informe de serie1; Ghidra en native/serie1/ghidra).
//
// What the game waits for (Ghidra, projelf_par1; native/serie1/ghidra):
//   - 0xa0de50 sets the view's labels (*(view+0x120)+0x120 / +0x1c0) to 0 and records the job "XPRPreRender" with
//     0xa04ed0(flags 2): a GRAPHICS context (0x9035e0), submitted by the next KickoffSubmitJob (0xa08ae0 -> SubmitJob
//     0xa18840 -> 0x911c90 CRenderContext::Submit = sce::Agc::submitGraphics): queue 0, the DCB, not an ACB. 0xa12350 kicks
//     it off first thing and then records RenderDepth / RenderGBuffer, whose passes (0xab6080) wait on the CPU for the
//     label (0x8351b0) and read the bits (0xa1f290).
//   - The emulator translates queue 0 in order (one command processor, FIFO): the culling is translated when everything
//     of frame N before it is. CanProcessSubmission / frame_epoch never hold it (they hold compute queues only); queue
//     0's FIFO does. The label is written at translation (WriteAtEndOfPipe); the game's first read of the bits then
//     waits for the GPU to RUN the dispatch (parpadeo6's settle -> readback -> the tick's fence), and on the one Vulkan
//     queue the dispatch runs after all the graphics work translated before it (frame N's tail).
//   - Log 20261116-093502-352: the GPU thread idles 2.1-2.3 ms a frame (the [render] "idle"); the game's settles wait
//     ~1.3-1.5 ms each for their readback ([parpadeo6] "leidos de vuelta": ~3,500-4,500 per 5 s).
// What this does (KYTY_SERIE1=1): a queue-0 command buffer with dispatches, a label write and no draws, no indirect
// buffers, no flip (the culling's shape) is TRANSLATED IN ITS PLACE as before (same command processor, same order: the
// caches, the labels and the waits see exactly what they saw), but recorded into a command buffer of the second Vulkan
// queue (lane 1 of local/lane-sync.h, KYTY_ASYNC_COMPUTE's machinery). That buffer waits by timeline semaphore for the
// graphics ticks that last wrote (or, for its writes, used) what it touches - the depth / HiZ of the frame before
// included - and not for the rest; every later graphics submission waits for it (the lanes' safe mode). The copy that
// reads its bits back goes on its lane too (KYTY_ACQ_READBACK_LANE's rule). No CPU wait is added on the GPU thread.
// KYTY_SERIE1=2: the same measurements, nothing moved (the "before").
// Line "[serie1]" every 5 s (the GPU thread).

#include "graphics/guest_gpu/pm4.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>

namespace Serie1 {

// 0 off, 1 on, 2 measure only. Read once.
[[nodiscard]] inline int Mode() noexcept {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_SERIE1");
		if (v == nullptr || v[0] == '\0' || v[1] != '\0') return 0;
		return v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
	}();
	return mode;
}
// The second Vulkan queue and the lanes are wanted (KYTY_SERIE1=1).
[[nodiscard]] inline bool WantsLanes() noexcept { return Mode() == 1; }
// KYTY_ASYNC_COMPUTE=1/2 (the game's compute queues on lane 1 too).
[[nodiscard]] inline bool AsyncComputeEnv() noexcept {
	static const bool on = [] {
		const char* a = std::getenv("KYTY_ASYNC_COMPUTE");
		return a != nullptr && (a[0] == '1' || a[0] == '2');
	}();
	return on;
}

inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

enum Kind : uint8_t { Unknown = 0, Cull = 1, NoDispatch = 2, Draws = 3, Indirect = 4, Flip = 5, NoLabel = 6, Malformed = 7 };

// A queue-0 command buffer's shape (top level; nothing it points at is read). Cull: dispatches, a label write (RELEASE_MEM
// / EOP / WRITE_DATA, real or the HLE's NOP-coded), and no draw, indirect buffer, predication or flip.
[[nodiscard]] inline Kind Classify(std::span<const uint32_t> commands) noexcept {
	namespace Pm4 = Libs::Graphics::Pm4;
	const uint32_t* p          = commands.data();
	const uint64_t  n          = commands.size();
	uint64_t        at         = 0;
	uint32_t        dispatches = 0;
	bool            label      = false;
	while (at < n) {
		const uint32_t header = p[at];
		if (header == 0x80000000u) { // (type 2 filler)
			++at;
			continue;
		}
		if ((header >> 30u) != 3u) return Malformed;
		const uint32_t len = KYTY_PM4_LEN(header);
		if (len > n - at) return Malformed;
		const uint32_t op = (header >> 8u) & 0xffu;
		switch (op) {
			case Pm4::IT_DISPATCH_DIRECT:
			case Pm4::IT_DISPATCH_INDIRECT: ++dispatches; break;
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_AUTO:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
			case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
			case Pm4::IT_DISPATCH_DRAW: return Draws;
			case Pm4::IT_INDIRECT_BUFFER:
			case Pm4::IT_INDIRECT_BUFFER_CNST:
			case Pm4::IT_COND_EXEC:
			case Pm4::IT_SET_PREDICATION:
			case Pm4::IT_REWIND: return Indirect;
			case Pm4::IT_RELEASE_MEM:
			case Pm4::IT_EVENT_WRITE_EOP:
			case Pm4::IT_WRITE_DATA: label = true; break;
			case Pm4::IT_NOP: {
				const uint32_t r = KYTY_PM4_R(header);
				if (r == Pm4::R_FLIP || r == Pm4::R_WAIT_FLIP_DONE || r == Pm4::R_SUPER_RESOLUTION) return Flip;
				if (r == Pm4::R_RELEASE_MEM || r == Pm4::R_WRITE_DATA) label = true;
				break;
			}
			default: break;
		}
		at += len;
	}
	if (dispatches == 0) return NoDispatch;
	return label ? Cull : NoLabel;
}

struct Counters {
	// GPU thread only (plain), but read by the report on the same thread.
	uint64_t classified[8] {};      // queue-0 graphics command buffers by Kind
	uint64_t culls_lane1 = 0;       // ... Cull ones recorded on lane 1
	uint64_t no_lane      = 0;       // ... Cull ones left on lane 0 (no second queue / KYTY_PAR_QUEUES / mode 2)
	// Lane-1 submissions of a culling (CommandScheduler::AddLaneWaits): what it had to wait for of the graphics lane.
	uint64_t avoided   = 0; // ESPERAS EVITADAS: graphics work was still pending and the culling did not wait for its newest
	uint64_t dependent = 0; // it needed the newest graphics tick (no overtaking: a real dependency)
	uint64_t idle_gfx  = 0; // the graphics lane had nothing pending (nothing to overtake)
	uint64_t overtaken = 0; // ticks of graphics work pending and not waited for (sum over the avoided)
	// The game's wait for the label: the culling's command buffer from queued (the game's passes start waiting right
	// after their KickoffSubmitJob queues it) to translated (its label written).
	uint64_t label_n = 0, label_ns = 0, label_max_ns = 0;
	// The GPU thread's gap after the label: to the next queue-0 graphics command buffer it starts.
	uint64_t gap_n = 0, gap_ns = 0;
	int64_t  gap_from_ns = 0; // the last culling's label (0: taken)
	// Label -> the game's last settle that read back its bits (the wait for the GPU to run the culling), per label.
	uint64_t bits_n = 0, bits_ns = 0, bits_max_ns = 0;
	uint64_t frames = 0;
	// Its reads through the page table (GpuResourceManager): bounded, noted per 64 KiB; unbounded: it waited for all.
	uint64_t bda_ranges = 0, bda_full = 0;
};
inline Counters g_c;
inline int64_t  g_last_report_ns = 0;
// (GPU thread) The open lane-1 command buffer holds a culling: its submissions are counted (AddLaneWaits).
inline bool g_open_cull = false;
// Set by the GPU thread at each culling label; the game's settles (any thread) keep the longest label -> bytes-in.
inline std::atomic<int64_t> g_label_ns {0};
inline std::atomic<int64_t> g_bits_max_ns {0};

// GPU thread: a culling's label was just translated (its command buffer complete).
inline void NoteLabel(int64_t enqueue_ns) noexcept {
	const int64_t now = NowNs();
	auto&         c   = g_c;
	if (enqueue_ns > 0 && now > enqueue_ns) {
		const auto d = static_cast<uint64_t>(now - enqueue_ns);
		++c.label_n;
		c.label_ns += d;
		c.label_max_ns = std::max(c.label_max_ns, d);
	}
	// The previous label's bits wait, folded.
	if (const int64_t b = g_bits_max_ns.exchange(0, std::memory_order_relaxed); b > 0) {
		++c.bits_n;
		c.bits_ns += static_cast<uint64_t>(b);
		c.bits_max_ns = std::max(c.bits_max_ns, static_cast<uint64_t>(b));
	}
	g_label_ns.store(now, std::memory_order_relaxed);
	c.gap_from_ns = now;
}
// GPU thread: a queue-0 graphics command buffer that is not a culling begins.
inline void NoteGraphicsStart() noexcept {
	auto& c = g_c;
	if (c.gap_from_ns == 0) return;
	const int64_t now = NowNs();
	if (now > c.gap_from_ns) {
		++c.gap_n;
		c.gap_ns += static_cast<uint64_t>(now - c.gap_from_ns);
	}
	c.gap_from_ns = 0;
}
// A game thread: a settle of XPR bits that read back GPU bytes ended now (0xa1f290's hook, parpadeo6).
inline void NoteSettleReadback() noexcept {
	const int64_t label = g_label_ns.load(std::memory_order_relaxed);
	if (label == 0) return;
	const int64_t d = NowNs() - label;
	if (d <= 0 || d > 100'000'000) return; // (another frame's, or a stall: not this label's)
	int64_t cur = g_bits_max_ns.load(std::memory_order_relaxed);
	while (d > cur && !g_bits_max_ns.compare_exchange_weak(cur, d, std::memory_order_relaxed)) {
	}
}
// GPU thread (AddLaneWaits): a lane-1 submission of a culling. `need`: the graphics tick it waits for (0: none);
// `newest`: the graphics lane's last tick; `newest_done`: the GPU finished it already.
inline void NoteCullSubmit(uint64_t need, uint64_t newest, bool newest_done) noexcept {
	auto& c = g_c;
	if (newest == 0 || newest_done) {
		++c.idle_gfx;
	} else if (need >= newest) {
		++c.dependent;
	} else {
		++c.avoided;
		c.overtaken += newest - need;
	}
}

// GPU thread, between submissions: the line every 5 s.
inline void MaybeReport(bool lanes_on) noexcept {
	const int64_t now = NowNs();
	if (g_last_report_ns == 0) {
		g_last_report_ns = now;
		return;
	}
	if (now - g_last_report_ns < 5'000'000'000ll) return;
	const double seconds = static_cast<double>(now - g_last_report_ns) / 1e9;
	g_last_report_ns     = now;
	auto&        c       = g_c;
	const double frames  = c.frames != 0 ? static_cast<double>(c.frames) : 1.0;
	const auto   ms      = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	const auto   avg     = [&](uint64_t ns, uint64_t count) { return count != 0 ? ms(ns) / static_cast<double>(count) : 0.0; };
	const auto   cull    = c.classified[Cull];
	std::printf("[serie1] %.1f s, %llu frames (KYTY_SERIE1=%d%s): culls del XPR (DCB solo de computo con etiqueta) %llu (%.2f/frame), "
	            "a la 2a cola %llu, sin mover %llu; otros DCB: con draws %llu, con IB/predicado %llu, flip %llu, sin dispatch %llu, "
	            "sin etiqueta %llu, raros %llu | ESPERAS EVITADAS %llu (el cull no espero al trabajo grafico pendiente: %.1f ticks de "
	            "media adelantados), SIN ADELANTO POR DEPENDENCIA %llu (necesitaba el ultimo trabajo grafico), nada pendiente %llu; lecturas por puntero del cull: con rango %llu, sin rango %llu (esperan a todo) | "
	            "ESPERA DEL JUEGO POR LA ETIQUETA (cull en cola -> traducido): media %.3f ms, max %.3f ms, %.3f ms/frame | "
	            "etiqueta -> bits dentro (espera a que la GPU corra el cull, asentados con readback): media %.3f ms, max %.3f ms, "
	            "%.3f ms/frame | hueco del hilo GPU tras la etiqueta: media %.3f ms, %.3f ms/frame\n",
	            seconds, static_cast<unsigned long long>(c.frames), Mode(),
	            Mode() == 1 ? (lanes_on ? ", cull en la 2a cola" : ", SIN 2a COLA: solo mide") : ", solo mide",
	            static_cast<unsigned long long>(cull), static_cast<double>(cull) / frames,
	            static_cast<unsigned long long>(c.culls_lane1), static_cast<unsigned long long>(c.no_lane),
	            static_cast<unsigned long long>(c.classified[Draws]), static_cast<unsigned long long>(c.classified[Indirect]),
	            static_cast<unsigned long long>(c.classified[Flip]), static_cast<unsigned long long>(c.classified[NoDispatch]),
	            static_cast<unsigned long long>(c.classified[NoLabel]), static_cast<unsigned long long>(c.classified[Malformed]),
	            static_cast<unsigned long long>(c.avoided),
	            c.avoided != 0 ? static_cast<double>(c.overtaken) / static_cast<double>(c.avoided) : 0.0,
	            static_cast<unsigned long long>(c.dependent), static_cast<unsigned long long>(c.idle_gfx),
	            static_cast<unsigned long long>(c.bda_ranges), static_cast<unsigned long long>(c.bda_full),
	            avg(c.label_ns, c.label_n), ms(c.label_max_ns), ms(c.label_ns) / frames,
	            avg(c.bits_ns, c.bits_n), ms(c.bits_max_ns), ms(c.bits_ns) / frames,
	            avg(c.gap_ns, c.gap_n), ms(c.gap_ns) / frames);
	std::fflush(stdout);
	const int64_t gap_from = c.gap_from_ns;
	c                      = Counters {};
	c.gap_from_ns          = gap_from;
}

} // namespace Serie1
