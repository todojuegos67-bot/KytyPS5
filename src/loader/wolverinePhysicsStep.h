#ifndef EMULATOR_SRC_LOADER_WOLVERINEPHYSICSSTEP_H_
#define EMULATOR_SRC_LOADER_WOLVERINEPHYSICSSTEP_H_

#include <xbyak/xbyak.h>

#include <array>
#include <cstdint>

namespace Loader {

struct Program;

// Marvel's Wolverine 1.05 (PPSA03671) only: Havok steps by the real frame time instead of a fixed 1/60 s.
//
// The physics system (global at 0xBDC26E0) hands Havok one step a game frame of `phys` (0xBDC26D0) seconds
// (sub_3E67A90). The time-scale manager (sub_409E2A0) sets phys = scale * ref, ref (0xBDC26D4) = 1 / the mode's
// target rate (1/60 in Performance and Pro), and marks it with a pointer in 0xBDC26D8. Animation and the rest of the
// game advance by the frame step (0xBEE5F80, the real frame time capped at 1/15 s). So below the target, physics falls
// behind wall time (debris floats: at 20 fps it runs at a third of real speed) while animation-driven bodies get
// velocities for a longer frame than physics steps (thrown bodies fly). On a PS5 at a locked 60 the two are equal.
//
// The hook replaces sub_3E87D40's choice of phys (0x3E87DBF, where xmm0 = phys and xmm1 = (float) frame step):
//   no manager (0xBDC26D8 == 0): phys = max(step, 1/90)            (the game's own code, unchanged)
//   manager:                     phys = manager_phys * max(step, ref) / ref
// manager_phys is the last value the manager stored (the hook keeps it beside its code, as phys now changes every
// frame). So at the target rate nothing changes, the manager's slow motion still applies, and below the target
// physics keeps real time in steps of at most the game's own 1/15 s cap.
namespace WolverinePhysicsStep {

constexpr uint64_t PATCH_VA  = 0x3E87DBF; // cmp qword [0xBDC26D8], 0 ; jne 0x3E87DD9 ; vmaxss ; vmovss [phys], xmm0
constexpr uint64_t RESUME_VA = 0x3E87DD9;
constexpr uint64_t PHYS_VA   = 0xBDC26D0;
constexpr uint64_t REF_VA    = 0xBDC26D4;
constexpr uint64_t OVR_VA    = 0xBDC26D8;
constexpr uint64_t MIN_VA    = 0x77E5B5C; // the game's float 1/90

constexpr std::array<uint8_t, 26> ORIGINAL = {0x48, 0x83, 0x3d, 0x11, 0xa9, 0xf3, 0x07, 0x00, 0x75,
                                               0x10, 0xc5, 0xf2, 0x5f, 0x05, 0x8b, 0xdd, 0x95, 0x03,
                                               0xc5, 0xfa, 0x11, 0x05, 0xf7, 0xa8, 0xf3, 0x07};

struct Addresses {
	uint64_t phys;
	uint64_t ref;
	uint64_t ovr;
	uint64_t min_step;
	uint64_t state; // two floats: the phys this hook last stored, the manager's last phys
};

// The hook's code: in xmm0 = phys and xmm1 = frame step, out xmm0 = phys (also stored). Uses rax, rcx, xmm2, xmm3,
// which sub_3E87D40 sets before reading after RESUME_VA. Ends with a jump to `resume`, or a ret when it is null (tests).
inline void EmitPhysicsStep(Xbyak::CodeGenerator& g, const Addresses& a, const void* resume) {
	using namespace Xbyak::util;
	Xbyak::Label scaled;
	Xbyak::Label keep;
	Xbyak::Label store;
	g.mov(rax, a.ovr);
	g.cmp(qword[rax], 0);
	g.jne(scaled);
	g.mov(rax, a.min_step);
	g.vmaxss(xmm0, xmm1, dword[rax]);
	g.jmp(store);
	g.L(scaled);
	g.mov(rax, a.state);
	g.vucomiss(xmm0, dword[rax]); // phys is still ours (or NaN): keep the manager's last value
	g.je(keep);
	g.vmovss(dword[rax + 4], xmm0);
	g.L(keep);
	g.mov(rcx, a.ref);
	g.vmovss(xmm2, dword[rcx]);
	g.vmaxss(xmm3, xmm1, xmm2);
	g.vdivss(xmm3, xmm3, xmm2);
	g.vmulss(xmm0, xmm3, dword[rax + 4]);
	g.L(store);
	g.mov(rax, a.state);
	g.vmovss(dword[rax], xmm0);
	g.mov(rax, a.phys);
	g.vmovss(dword[rax], xmm0);
	if (resume != nullptr) {
		g.jmp(resume, Xbyak::CodeGenerator::T_NEAR);
	} else {
		g.ret();
	}
}

} // namespace WolverinePhysicsStep

// Installs the hook when the main program is the 1.05 eboot and the title hook area exists (it is the end of the
// instruction trampoline area, so --redzone, --amd-cpu or --guest-tsc); KYTY_WOLVERINE_PHYSICS_STEP=0 leaves the game's
// fixed step. Call after relocation, before guest code runs.
void InstallWolverinePhysicsStep(Program* main_program);

} // namespace Loader

#endif /* EMULATOR_SRC_LOADER_WOLVERINEPHYSICSSTEP_H_ */
