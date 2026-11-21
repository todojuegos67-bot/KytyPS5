#pragma once
// (f5) Per-pass verification (=2): the interface (design; no implementation in this build).
//
// One play session must validate many passes at once. Every converted pass has its own switch (KYTY_NATIVE_PASSES
// "Name:2"). In Verify the native pass writes shadow targets; at the scope end the GPU compares them with the emulated
// pass's targets (the comparison compute of render1 F2, render1-cmp.comp: 0 of 5.792.373 pixels differed there) and
// the state the emulated range left with the state the native pass declares. Results are read back without waiting,
// a few frames later, and logged per pass:
//   [f5] VERIFICA "<pass>" frames=<n> pixeles=<n> distintos=<n> max_delta=<d> estado_distinto=<n>
//   [f5] DISCREPANCIA "<pass>" frame=<f> objetivo=<i> distintos=<n> (first ones, rate-limited)
// A pass is promoted to Native only after a soak with 0 DISCREPANCIA (bit-exact targets for depth / geometry passes;
// a declared tolerance for passes whose emulated output is itself not deterministic, e.g. atomics order).
#include <cstdint>

namespace NativeRender {

struct VerifyTarget {
	uint64_t emulated_image = 0; // the game's target as the emulated path wrote it
	uint64_t shadow_image   = 0; // the native pass's copy
	uint32_t format = 0, width = 0, height = 0;
	uint32_t tolerance = 0;      // max per-channel difference accepted (0 = bit exact)
};

struct VerifyResult {
	uint64_t frame        = 0;
	uint32_t pass         = 0;
	uint64_t pixels       = 0;
	uint64_t differing    = 0;
	uint32_t max_delta    = 0;
	uint32_t state_differs = 0; // registers / user data words that differ at the scope end
};

class PassVerifier {
public:
	virtual ~PassVerifier() = default;
	// Record the comparison of `count` targets at the pass's scope end (GPU thread, the open command buffer).
	virtual void Compare(uint32_t pass, const VerifyTarget* targets, uint32_t count, void* cmd) = 0;
	// Results that are ready (never waits); logs the lines above.
	virtual void Collect() = 0;
};

} // namespace NativeRender
