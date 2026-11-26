#ifndef EMULATOR_SRC_LOCAL_PORT3_H_
#define EMULATOR_SRC_LOCAL_PORT3_H_
// (port3) KYTY_PORT3=1|2 (off by default): stage E2b of the native-port plan, on top of port2 (local/port2.h). PORT3
// implies PORT2: with KYTY_PORT3 set, Port2::Mode() is PORT3's mode whatever KYTY_PORT2 says.
//
// What it adds to port2's flow (each part measured against the [port2] / [otros1] lines of the PC logs,
// port3/INFORME.md):
//   A. Steps in place: a span that holds an EVENT_WRITE (0x46), a WAIT_REG_MEM (0x3c / 0x3d), an ACQUIRE_MEM or
//      RELEASE_MEM NOP, a push / pop marker or a plain NOP is no longer cut. The flow records the step's position and
//      header only; the command processor runs that packet in its place in the flow, from the command buffer itself
//      (what a later patch of the game wrote is read, as the walk reads it: AgcWaitRegMemPatchAddress / Reference),
//      with the walk's own work around it (the guest commands and copy feedback before it, the deferred protections
//      flushed, the draw state observer, the same handler). A wait that is not satisfied suspends the walk at that
//      packet (the walk takes it again when it resumes, then the rest of the span as PM4); a header that is not the
//      recorded one ends the flow there (the walk goes on from that packet).
//   B. Prefix: a span with any other packet (CONTEXT_STATE, the linear copy's announcement, a predicated packet, ...)
//      is sealed up to that packet (the flow covers what lies before it, the walk the rest); a span longer than 1024 dw
//      up to there; the span after a command buffer's last draw is sealed at its submit when its marker lies in the
//      submitted range (before the GPU is given it; the other pending markers are dropped as in port2).
//   C. The command processor: (1) the step right after a port2 marker that lands on a draw or a dispatch does not run
//      the walk's guest commands / copy feedback again (the marker's step ran them < 1 us before: nothing but register
//      state came between, like a register lane of render5); (2) the draw before a sealed marker prefetches the marker's
//      record (cold: a game thread wrote it), as PrefetchNextDraw does for the next draw's tables.
// Modes: 1: alternating 5 s ON / 5 s OFF (OFF: port2 and port3 off, the stream is today's; KYTY_PORT3_CONTRA=2: OFF is
// port2 alone, so AHORRO MEDIDO is port3's own share) and a [port3] line every 5 s with AHORRO MEDIDO (averages and
// paired windows). 2: verification (port2's, plus the steps: each recorded step's header and offset against the PM4,
// the image is the PM4's). KYTY_PORT3_FIJO=1: always on (mode 1 without alternation), a short line every 30 s.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Port3 {

[[nodiscard]] inline uint32_t EnvU(const char* name) {
	const char* text = std::getenv(name);
	return text == nullptr || text[0] == '\0' ? 0u : static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
}
// KYTY_PORT3_FIJO=1: on for good (mode 1, no alternation).
[[nodiscard]] inline bool Fixed() {
	static const bool fixed = EnvU("KYTY_PORT3_FIJO") == 1;
	return fixed;
}
[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		uint32_t m = EnvU("KYTY_PORT3");
		if (m > 2) m = 0;
		if (m == 0 && Fixed()) m = 1;
		if (m == 2 && Fixed()) m = 1; // (fixed is a playing mode: no verification)
		if (m != 0)
			std::printf("port3: KYTY_PORT3=%u%s: port2's flow without its cuts (steps in place, prefixes, submit) and the "
			            "command processor's walk work after a marker%s\n",
			            m, Fixed() ? " (FIJO)" : "",
			            m == 2 ? " - verification: flow and PM4 both applied and compared, the image is the PM4's"
			            : Fixed() ? " - always on"
			                      : " - alternating 5 s ON / 5 s OFF, [port3] line every 5 s");
		return m;
	}();
	return mode;
}
// KYTY_PORT3_CONTRA=2: the OFF windows run port2 alone (AHORRO MEDIDO = port3's own share); else nothing.
[[nodiscard]] inline uint32_t Against() {
	static const uint32_t against = EnvU("KYTY_PORT3_CONTRA") == 2 ? 2u : 0u;
	return against;
}
inline std::atomic<bool> g_on {true}; // (1) port3's ON window; flipped at the [port3] line
// Port3's recording side (steps, prefixes, submit) in this window.
[[nodiscard]] inline bool Active() { return Mode() != 0 && g_on.load(std::memory_order_relaxed); }
// The command processor's part (C): mode 1 only (2 keeps the walk's work exactly as it is).
[[nodiscard]] inline bool GpuActive() { return Mode() == 1 && g_on.load(std::memory_order_relaxed); }

// ---- counters ----
// Steps in place by kind (recorded / run).
enum Step : uint32_t { SEvent, SWait, SRelease, SAcquire, SNop, Steps };
inline constexpr const char* StepNames[Steps] = {"evento", "espera", "release_mem", "acquire_mem", "nop"};
// Why a prefix ended (the packet the walk takes from there).
enum Cut : uint32_t { COpcode, CNop, CHu, CPred, CShNop, CLen, CBig, CLong, Cuts };
inline constexpr const char* CutNames[Cuts] = {"otro_paquete", "nop_otro", "marcador_hu_otro", "predicado", "sh_nop",
                                               "longitud", "flujo_grande", "corte_de_tramo"};
inline std::atomic<uint64_t> g_rec_step[Steps] {}, g_rec_prefix[Cuts] {}, g_rec_submit_sealed {0}, g_rec_long_sealed {0};
inline std::atomic<uint64_t> g_rec_nop_r[64] {}; // (the R type of the NOPs that end a prefix or a span: not taken)
inline std::atomic<uint64_t> g_gpu_step[Steps] {}, g_gpu_wait_suspended {0}, g_gpu_step_header {0}, g_gpu_pre_skipped {0},
    g_gpu_prefetched {0};
// (2) The steps: recorded header / offset against the PM4 met there.
inline std::atomic<uint64_t> g_step_mismatch {0}, g_partial {0};

} // namespace Port3

#endif // EMULATOR_SRC_LOCAL_PORT3_H_
