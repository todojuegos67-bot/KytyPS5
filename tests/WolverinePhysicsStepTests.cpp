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
float    g_ref      = 1.0f / 60.0f;
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
	    {reinterpret_cast<uint64_t>(&g_phys), reinterpret_cast<uint64_t>(&g_ref), reinterpret_cast<uint64_t>(&g_ovr),
	     reinterpret_cast<uint64_t>(&g_min), reinterpret_cast<uint64_t>(g_state)},
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

	// The manager's slow motion (scale 0.5) still applies, on top of the real frame time.
	g_phys = 0.5f / 60.0f;
	Check(Near(Frame(hook, 0.05f), 0.025f), "slow motion at 20 fps: not 0.5 * 0.05");
	Check(Near(Frame(hook, 0.04f), 0.02f), "slow motion kept: not 0.5 * 0.04");

	// A mode change stores phys = ref (1/30 in Fidelity).
	g_ref  = 1.0f / 30.0f;
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
