#pragma once
// (g1) "Lo gordo nativo", one build that measures the two largest GPU-thread costs left (S1, M1) and the per-draw walk
// NP1 found, and implements the indirect-state delta behind a switch. Demon's Souls PS5 01.007.000 only.
//
// Switches (read once; all off by default; off = one relaxed load per hook):
//   KYTY_G1_PROBE=1  measurement. "[g1]" lines every 5 s and "[g1] RESUMEN" every 60 s:
//     A) indirect register state. Game side: the binder's memcpy of register pairs into the block it keeps at the top
//        of the command buffer (Ghidra, 01.007.000: 0x1d182f0 CX / 0x1d18460 SH / 0x1d185d0 UC, each
//        memcpy(block + used, source, n * 8) then PatchAddRegisters; the call returns to 0x1d18434 / 0x1d185a4 /
//        0x1d18714). Per copy: the source (eboot static / stack / heap by memory-map label), whether the source was
//        copied before (this or the previous frame) and whether its bytes are the same (XXH3 of the n * 8 bytes).
//        GPU side (CpOpIndirect*Regs): blocks and pairs, the time of the block's sync, of its apply and in all, the pairs
//        by kind (proven raw store / proven ignored / handler call / fake / skipped), the true delta (raw pairs whose
//        stored value differs: exact; call pairs whose value differs from the last one an indirect block wrote: the
//        graphics context reaches the GPU only through these blocks), blocks whose bytes were applied before.
//     B) coherence of GPU-written memory on the GPU thread: read-flush / write-release / read-track / write-track
//        protection changes by memory-map label (exact address) and by the frame map's innermost pass (KYTY_NATIVE_RENDER
//        =map), the pages GPU work marked no-access by pass and label, tq6 (table dispatch phase 6, KYTY_RENDER_STATS) by
//        pass, and the CPU's reads of GPU-written pages (read faults the GPU resources handled) by call site: the known
//        readers 0xa1f560, 0xa1f290 (-> 0xad8bc0 / 0xad92f0), 0xa1f020, the accessors 0x909940 / 0x909970, other eboot
//        sites (top offsets) and code outside the eboot. TECHO B: the GPU-thread ms of labels all of whose reads come
//        from known readers.
//     NP1) the general path's walk of a draw packet outside the table try, split by stretch (preamble, decode, the
//        walk's checks, the entry, the state observer, the try, after it), and the table set checks: how many proofs
//        held, were checked again, refreshed, failed; reproved ranges that were clean or not.
//   KYTY_G1_STATE=1|2  the indirect-state delta (native/state-cache.inc, in pm4Handlers.cpp): a block's decisions kept by
//     its content (XXH3 of its pairs + their bytes compared), proven raw stores made only when the stored value differs,
//     proven-ignored and fake registers not called, the rest called as before. 1: 5 s ON / 5 s OFF with "[g1] ALTERNA"
//     (paired OFF -> ON windows, AHORRO MEDIDO); KYTY_G1_FIJO=1 keeps it ON. 2: verification, both paths from the same
//     registers, the context / shader / user-config register files compared (DISCREPANCIA must be 0); the normal path's
//     result is kept.
// The coherence part (KYTY_G1_COH) is NOT implemented in this build: see INFORME (the probe decides it).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <x86intrin.h>

namespace G1 {

struct Switches {
	uint32_t probe = 0;
	uint32_t state = 0;
	bool     fixed = false;
};
[[nodiscard]] Switches ReadSwitches() noexcept;
[[nodiscard]] inline const Switches& Get() noexcept {
	static const Switches s = ReadSwitches();
	return s;
}
[[nodiscard]] inline uint32_t ProbeMode() noexcept { return Get().probe; }
[[nodiscard]] inline uint32_t StateMode() noexcept { return Get().state; }
[[nodiscard]] inline bool     On() noexcept { return Get().probe != 0 || Get().state != 0; }

// The GPU (render) thread: set at its first flip with G1 on.
inline thread_local bool t_gpu = false;
inline std::atomic<uint64_t> g_frame {1}; // flips seen (GPU thread writes)
// (STATE=1) the ON window (flipped every 5 s unless KYTY_G1_FIJO=1).
inline std::atomic<bool> g_window_on {true};
[[nodiscard]] inline bool StateActive() noexcept {
	const uint32_t m = StateMode();
	return m == 2 || (m == 1 && g_window_on.load(std::memory_order_relaxed));
}
// For F0's frame split (live-control.h): 0 no alternation, 1 ON window, 2 OFF window.
[[nodiscard]] inline int AbWindow() noexcept {
	return StateMode() == 1 && !Get().fixed ? (g_window_on.load(std::memory_order_relaxed) ? 1 : 2) : 0;
}

// ---- loader (loader/demonsSoulsCopy.cpp Install): the eboot's binder copies checked, its base kept ----
void Install(uint64_t base, uint64_t size, bool is_eboot) noexcept;

// ---- game threads: libc memcpy (libs/libC.cpp) ----
inline std::atomic<uint64_t> g_copy_lo {0}; // eboot + 0x1d18434 once checked with the probe on; 0 otherwise
constexpr uint64_t           CopySpan = 0x1d18714u - 0x1d18434u + 1u;
inline thread_local uint64_t t_memcpy_ret = 0; // (B) the guest caller of the host memcpy running now
void OnBinderCopySlow(uint64_t ret, const void* dst, const void* src, size_t n) noexcept;
inline void OnMemcpy(uint64_t ret, const void* dst, const void* src, size_t n) noexcept {
	const uint64_t lo = g_copy_lo.load(std::memory_order_relaxed);
	if (lo == 0) return;
	t_memcpy_ret = ret;
	if (ret - lo < CopySpan) OnBinderCopySlow(ret, dst, src, n);
}
inline void AfterMemcpy() noexcept {
	if (g_copy_lo.load(std::memory_order_relaxed) != 0) t_memcpy_ret = 0;
}

// ---- GPU thread: indirect register blocks (pm4Handlers.cpp through state-cache.inc) ----
// Plain counters, the GPU thread's only (other threads' blocks are counted in g_other_blocks).
struct BlockCounters {
	uint64_t blocks[3] {}, pairs[3] {};
	uint64_t cyc_sync[3] {}, cyc_apply[3] {}, cyc_total[3] {};
	uint64_t raw[3] {}, raw_diff[3] {}, ign[3] {}, call[3] {}, call_diff[3] {}, fake[3] {}, skipped[3] {};
	uint64_t repeat_blocks[3] {}, repeat_pairs[3] {}, same_as_previous[3] {};
	uint64_t probe_cycles = 0;
	// STATE
	uint64_t st_hit[3] {}, st_miss[3] {}, st_stores[3] {}, st_kept[3] {}, st_calls[3] {}, st_not_called[3] {};
	uint64_t st_verified = 0, st_discrepancias = 0;
};
inline BlockCounters         g_blocks;
inline std::atomic<uint64_t> g_other_blocks {0};

// ---- B: coherence (any thread) ----
inline std::atomic<bool> g_coh {false}; // the probe on and the eboot checked
// memoryAddressSpace.inc, a protection change of guest memory: `origin` Raiz::ProtectOrigin.
void NoteProtectSlow(bool gpu, uint32_t origin, uint64_t vaddr, uint64_t size, uint64_t ns) noexcept;
inline void NoteProtect(bool gpu, uint32_t origin, uint64_t vaddr, uint64_t size, uint64_t ns) noexcept {
	if (g_coh.load(std::memory_order_relaxed)) NoteProtectSlow(gpu, origin, vaddr, size, ns);
}
// pageManager.cpp AddReadProtect: GPU work marked [begin, end) no-access (deferred).
void NoteReadTrackSlow(uint64_t begin, uint64_t end) noexcept;
inline void NoteReadTrack(uint64_t begin, uint64_t end) noexcept {
	if (g_coh.load(std::memory_order_relaxed)) NoteReadTrackSlow(begin, end);
}
// live-census.h: a table dispatch's phase 6 (tq6) took `cycles`.
void NoteTq6Slow(uint64_t cycles) noexcept;
inline void NoteTq6(uint64_t cycles) noexcept {
	if (g_coh.load(std::memory_order_relaxed)) NoteTq6Slow(cycles);
}
// runtimeLinker.cpp: a fault on guest memory the GPU resources handled. `rip` the faulting instruction, `rsp` its stack.
void NoteGpuFaultSlow(bool read, uint64_t rip, uint64_t rsp, uint64_t vaddr) noexcept;
inline void NoteGpuFault(bool read, uint64_t rip, uint64_t rsp, uint64_t vaddr) noexcept {
	if (g_coh.load(std::memory_order_relaxed)) NoteGpuFaultSlow(read, rip, rsp, vaddr);
}

// ---- NP1: the walk of a draw packet (graphicsRun.cpp ProcessPm4), GPU thread ----
enum WalkPoint : uint32_t {
	WTop = 0,      // the step begins (pm4_clock.Next)
	WPreamble = 1, // the preamble done
	WDecoded = 2,  // header, class
	WChecked = 3,  // the walk's checks done (predication, lane, chunk, speculation, state pass, port2 marker, linear
	               // copy, deferred protection, techo, path tracing, native walk): the handler is looked up
	WEntry = 4,    // render1, prefetch, capture: the native XPR block begins
	WObserved = 5, // the draw state observer done: the try begins
	WTried = 6,    // the try returned
	WalkPoints = 7
};
inline std::atomic<bool> g_walk {false};
struct WalkState {
	uint64_t mark[WalkPoints] {};
	uint32_t set  = 0;
	bool     draw = false;
	// accumulated: [0] steps with a try, [1] without; by the stretch's first point
	uint64_t cyc[2][WalkPoints] {};
	uint64_t steps[2] {};
	uint64_t dropped = 0; // steps longer than 2 ms (the walk returned after the draw): not counted
};
inline WalkState g_walk_state;
void WalkStepSlow() noexcept;
inline void WalkStep() noexcept {
	if (g_walk.load(std::memory_order_relaxed) && t_gpu) WalkStepSlow();
}
inline void WalkMark(WalkPoint p) noexcept {
	if (g_walk.load(std::memory_order_relaxed) && t_gpu) {
		g_walk_state.mark[p] = __rdtsc();
		g_walk_state.set |= 1u << p;
	}
}
inline void WalkDecoded(bool draw) noexcept {
	if (g_walk.load(std::memory_order_relaxed) && t_gpu) {
		g_walk_state.mark[WDecoded] = __rdtsc();
		g_walk_state.set |= 1u << WDecoded;
		g_walk_state.draw = draw;
	}
}
// table-xpr.inc: a set's proofs looked at (TableValidateSet past its frame check), each image use held or checked,
// refreshed (streamed in), the outcome; a block range proved clean again (phase 6) or not.
struct SetCounters {
	std::atomic<uint64_t> calls {0}, full {0}, held {0}, checked {0}, refreshed {0}, ok {0};
	std::atomic<uint64_t> reprove_clean {0}, reprove_dirty {0};
};
inline SetCounters g_sets;
inline void NoteSetCheck(bool full) noexcept {
	if (!g_walk.load(std::memory_order_relaxed)) return;
	g_sets.calls.fetch_add(1, std::memory_order_relaxed);
	if (full) g_sets.full.fetch_add(1, std::memory_order_relaxed);
}
inline void NoteSetUse(bool held) noexcept {
	if (g_walk.load(std::memory_order_relaxed)) (held ? g_sets.held : g_sets.checked).fetch_add(1, std::memory_order_relaxed);
}
inline void NoteSetRefresh() noexcept {
	if (g_walk.load(std::memory_order_relaxed)) g_sets.refreshed.fetch_add(1, std::memory_order_relaxed);
}
inline void NoteSetOk() noexcept {
	if (g_walk.load(std::memory_order_relaxed)) g_sets.ok.fetch_add(1, std::memory_order_relaxed);
}
inline void NoteReprove(bool clean) noexcept {
	if (g_walk.load(std::memory_order_relaxed))
		(clean ? g_sets.reprove_clean : g_sets.reprove_dirty).fetch_add(1, std::memory_order_relaxed);
}

// ---- the flip (live-control.h, GPU thread) ----
void FlipSlow(int64_t flip_ns) noexcept;
inline void Flip(int64_t flip_ns) noexcept {
	if (On()) FlipSlow(flip_ns);
}

} // namespace G1
