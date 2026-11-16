#include "loader/wolverinePhysicsStep.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Loader::WolverinePhysicsStep;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WolverinePhysicsStepTests: failed: %s\n", message);
		std::abort();
	}
}

bool Near(float a, float b) {
	return std::fabs(a - b) <= 1e-6f * std::fmax(1.0f, std::fabs(b));
}

// The game's globals, as the hook sees them.
float    g_phys     = 0.0f;
uint64_t g_ovr      = 0;
float    g_min      = 1.0f / 90.0f;
float    g_state[2] = {};

using Hook = float (*)(float phys, float step); // xmm0, xmm1 in and xmm0 out, on both host ABIs

// One frame of sub_3E87D40: it loads phys, then the hook chooses the step Havok gets.
float Frame(Hook hook, float step) {
	return hook(g_phys, step);
}

void TestPhysicsStep() {
	Xbyak::CodeGenerator gen;
	Loader::WolverinePhysicsStep::EmitPhysicsStep(
	    gen,
	    {reinterpret_cast<uint64_t>(&g_phys), reinterpret_cast<uint64_t>(&g_ovr), reinterpret_cast<uint64_t>(&g_min),
	     reinterpret_cast<uint64_t>(g_state)},
	    nullptr);
	gen.ready();
	auto hook = gen.getCode<Hook>();

	// No manager: the game's own variable step, max(step, 1/90).
	g_phys = 1.0f / 60.0f;
	Check(Near(Frame(hook, 0.05f), 0.05f) && Near(g_phys, 0.05f), "no manager: not the frame step");
	Check(Near(Frame(hook, 0.005f), 1.0f / 90.0f), "no manager: not clamped to 1/90");

	// The manager stores scale * ref = 1/60 and marks itself.
	g_ovr  = 0x90bee5f38;
	g_phys = 1.0f / 60.0f;
	Check(Near(Frame(hook, 1.0f / 15.0f), 1.0f / 15.0f) && Near(g_phys, 1.0f / 15.0f), "15 fps: not 1/15");
	// Next frames read back the hook's own phys; the manager's 1/60 stays the base.
	Check(Near(Frame(hook, 0.05f), 0.05f), "20 fps: not 0.05 (the hook's own phys taken as the manager's)");
	Check(Near(Frame(hook, 1.0f / 60.0f), 1.0f / 60.0f), "at the target: not the game's 1/60");
	Check(Near(Frame(hook, 0.01f), 1.0f / 60.0f), "above the target: not the game's 1/60");

	// The manager's slow motion (scale 0.5): it stores 0.5 * ref, and the frame step arrives scaled too (0.5 * frame
	// time; 73-trask-fight, 2026-11-16: 0.0175 at a 0.035 s frame). Havok gets that step, not the scale twice.
	g_phys = 0.5f / 60.0f;
	Check(Near(Frame(hook, 0.0175f), 0.0175f), "slow motion at 28 fps: not the scaled step (0.5 * 0.035)");
	Check(Near(Frame(hook, 0.025f), 0.025f), "slow motion kept at 20 fps: not 0.5 * 0.05");
	Check(Near(Frame(hook, 0.5f / 60.0f), 0.5f / 60.0f), "slow motion at the target: not the game's 0.5 / 60");
	Check(Near(Frame(hook, 0.004f), 0.5f / 60.0f), "slow motion above the target: not the game's 0.5 / 60");
	// Deep slow motion (0.3) at 40 fps: the step (0.0075) is under ref but over the manager's 0.3 / 60.
	g_phys = 0.3f / 60.0f;
	Check(Near(Frame(hook, 0.0075f), 0.0075f), "deep slow motion at 40 fps: not the scaled step (the old fixed step)");

	// Time stopped by the manager (scale 0): it stores 0 and the scaled step is 0, so nothing moves.
	g_phys = 0.0f;
	Check(Frame(hook, 0.0f) == 0.0f && g_phys == 0.0f, "scale 0: physics advances");
	// A pause at scale 1 (the "L1 Parry" prompt in 73-trask-fight: step 0, pause flag set): the manager's 1/60 as before
	// this change; the game doesn't step Havok while the pause flag is set.
	g_phys = 1.0f / 60.0f;
	Check(Near(Frame(hook, 0.0f), 1.0f / 60.0f), "pause at scale 1: not the manager's 1/60");

	// A mode change stores phys = ref (1/30 in Fidelity).
	g_phys = 1.0f / 30.0f;
	Check(Near(Frame(hook, 0.05f), 0.05f), "1/30 mode at 20 fps: not 0.05");
	Check(Near(Frame(hook, 0.02f), 1.0f / 30.0f), "1/30 mode above its target: not 1/30");
}

} // namespace

int main() {
	TestPhysicsStep();
	std::printf("WolverinePhysicsStepTests: ok\n");
	return 0;
}
