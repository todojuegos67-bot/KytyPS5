#pragma once
// (envio3) KYTY_ENVIO3 (off by default; implies KYTY_ENVIO2 in the same mode, local/envio2.h). On top of ENVIO2:
//  - several compute submissions in flight on Kyty.Acb2 (a FIFO of at most MaxInFlight: one per compute queue head,
//    each committed by the GPU thread in its queue's order, into the one Vulkan queue, validated at the commit);
//  - continuations: a submission stopped at a WAIT_REG_MEM on memory the CPU writes (ENVIO2's "parcial": 100 % of its
//    stops, measured) or resumed at such a wait has the wait run on the GPU thread (its sweep of CPU writes included)
//    and the rest handed to Kyty.Acb2 again, instead of being translated the normal way on the GPU thread;
//  - the RAIZ race: what the GPU thread's sweeps invalidate as CPU-written while a submission is in flight is logged
//    (Spec::NoteHostWrite) and checked by its commit (clean reads stop, buffers re-uploaded, images stop).
// 2: verification (each committed at once, as ENVIO2's mode 2). A "[envio3]" line every 5 s next to "[envio2]".
#include "envio2.h"

namespace Envio3 {

inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_ENVIO3");
		if (value == nullptr || value[1] != '\0') return 0u;
		return value[0] == '1' ? 1u : value[0] == '2' ? 2u : 0u;
	}();
	return mode;
}

inline constexpr uint32_t MaxInFlight = 12;

struct Totals {
	std::atomic<uint64_t> continuations {0};    // continuations handed to Kyty.Acb2
	std::atomic<uint64_t> cont_unsatisfied {0}; // a continuation's wait not satisfied yet (the slice waits)
	std::atomic<uint64_t> cont_full {0};        // a continuation not handed: the FIFO full
	std::atomic<uint64_t> depth_sum {0}, depth_samples {0}, depth_max {0}; // in flight at each hand-off
	std::atomic<uint64_t> moved_ns {0};         // Kyty.Acb2's translation time whose work was committed (kept)
	std::atomic<uint64_t> lost_ns {0};          // its translation time thrown away (stops, changes)
	std::atomic<uint64_t> sweep_stops {0};      // commits stopped by what a sweep found CPU-written meanwhile
};
inline Totals g_totals;

inline void NoteDepth(uint64_t depth) {
	Envio2::Add(g_totals.depth_sum, depth);
	Envio2::Add(g_totals.depth_samples);
	auto seen = g_totals.depth_max.load(std::memory_order_relaxed);
	while (depth > seen && !g_totals.depth_max.compare_exchange_weak(seen, depth, std::memory_order_relaxed)) {}
}

// Printed by Envio2::NoteFrame's caller with the same frame count (`frames`).
inline void Print(double frames) {
	if (Mode() == 0) return;
	auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	const double f       = std::max(frames, 1.0);
	const auto   samples = take(g_totals.depth_samples);
	const auto   sum     = take(g_totals.depth_sum);
	std::printf("[envio3] mode %u: continuaciones a Acb2 %.1f/frame (espera no cumplida %.1f, cola llena %.1f); "
	            "profundidad de cola media %.1f max %llu (limite %u); ms reales movidos a Acb2 %.2f ms/frame "
	            "(traduccion tirada %.2f ms/frame); paradas por barrido RAIZ %.1f/frame\n",
	            Mode(), static_cast<double>(take(g_totals.continuations)) / f,
	            static_cast<double>(take(g_totals.cont_unsatisfied)) / f, static_cast<double>(take(g_totals.cont_full)) / f,
	            samples == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(samples),
	            static_cast<unsigned long long>(take(g_totals.depth_max)), MaxInFlight,
	            static_cast<double>(take(g_totals.moved_ns)) / 1e6 / f, static_cast<double>(take(g_totals.lost_ns)) / 1e6 / f,
	            static_cast<double>(take(g_totals.sweep_stops)) / f);
	std::fflush(stdout);
}

} // namespace Envio3
