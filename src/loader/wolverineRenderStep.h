#ifndef KYTY_LOADER_WOLVERINE_RENDER_STEP_H_
#define KYTY_LOADER_WOLVERINE_RENDER_STEP_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>

namespace Loader {

struct Program;

// Marvel's Wolverine 1.05 (PPSA03671, contentVersion 01.001.005) only: pins the game's dynamic-resolution step.
//
// The game renders its 3D at one of nine steps (the table at module offset 0xBEAB038, count at 0xBEAB128, filled at
// boot) and its own TAA scales that up to the 2560x1440 scan-out of Performance mode. Its controller, sub_3F02F10,
// runs once a frame and stores the step in 0x9B7BA54 and the step's render buffers in 0xBE800C0. It aims at the
// Performance mode's 60 fps, so below that it sits at the lowest step, 1664x936 (Wolverine Engine.md 8.3).
//
// The pin replaces the controller's first test (0x3F02FFF, "is dynamic resolution fixed?") with: step = the chosen
// index, clamped to the game's own count - 1, then a jump to the controller's own store (0x3F033C0). So the game keeps
// choosing nothing but the buffers, and every step's buffers already exist (sub_3F02760 makes them all at boot).
namespace WolverineRenderStep {

struct Step {
	uint32_t width;
	uint32_t height;
};

// The table at 0xBEAB038, as the game fills it (read from guest memory in six title scenes, 2026-10-22).
constexpr std::array<Step, 9> STEPS = {{{3328, 1872},
                                         {3072, 1728},
                                         {2816, 1584},
                                         {2560, 1440},
                                         {2304, 1296},
                                         {2048, 1152},
                                         {1920, 1080},
                                         {1792, 1008},
                                         {1664, 936}}};

constexpr uint64_t PATCH_VA  = 0x3F02FFF; // cmp byte [0xBEAB728], 0 ; jne 0x3F03016 ; ...
constexpr uint64_t STORE_VA  = 0x3F033C0; // cmp eax, [0x9B7BA54] ; je ; mov [0x9B7BA54], eax ; ... buffers
constexpr uint64_t COUNT_VA  = 0xBEAB128;
constexpr uint64_t TABLE_VA  = 0xBEAB038;
constexpr uint64_t ACTIVE_VA = 0x9B7BA54;

// The 1.05 bytes the patch replaces, and those of the store it jumps to: the build check.
constexpr std::array<uint8_t, 23> ORIGINAL = {0x80, 0x3d, 0x22, 0x87, 0xfa, 0x07, 0x00, 0x75, 0x0e, 0x8b, 0x35, 0x2e,
                                               0x87, 0xfa, 0x07, 0x3b, 0x35, 0x8c, 0x2f, 0xfe, 0x07, 0x75, 0x32};
constexpr std::array<uint8_t, 49> STORE    = {
    0x3b, 0x05, 0x8e, 0x86, 0xc7, 0x05, 0x74, 0x29, 0x89, 0x05, 0x86, 0x86, 0xc7, 0x05, 0x48, 0x98, 0x48,
    0x69, 0xc0, 0xf0, 0xc6, 0x01, 0x00, 0x48, 0x03, 0x05, 0x42, 0xc1, 0xf9, 0x07, 0x48, 0x89, 0x05, 0xdb,
    0xcc, 0xf7, 0x07, 0x8b, 0x05, 0x05, 0xad, 0xfa, 0x07, 0x89, 0x05, 0x63, 0xc1, 0xf9, 0x07};

// The same table in PS5 Pro mode (--trinity), where the game scales to a 3840x2160 scan-out: eleven steps (read from
// guest memory at the Pro forest title, 2026-11-11, session 58).
constexpr std::array<Step, 11> PRO_STEPS = {{{3840, 2160},
                                             {3584, 2016},
                                             {3328, 1872},
                                             {3072, 1728},
                                             {2816, 1584},
                                             {2560, 1440},
                                             {2304, 1296},
                                             {2048, 1152},
                                             {1920, 1080},
                                             {1792, 1008},
                                             {1664, 936}}};

// The size of step `index` in the table of the mode (Pro or base); {0, 0} past its end.
constexpr Step StepAt(bool pro, uint32_t index) {
	if (pro) {
		return index < PRO_STEPS.size() ? PRO_STEPS[index] : Step {0, 0};
	}
	return index < STEPS.size() ? STEPS[index] : Step {0, 0};
}

// The index of a step in the table of the mode (STEPS, or PRO_STEPS when `pro`), or -1.
constexpr int FindStep(uint32_t width, uint32_t height, bool pro = false) {
	for (uint32_t i = 0; StepAt(pro, i).width != 0; i++) {
		if (StepAt(pro, i).width == width && StepAt(pro, i).height == height) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

// DLSS 310.7 takes render sizes from 1920x1080 up at a 3840x2160 output, and the game drops to 1664x936; so with DLSS
// asked for in Pro mode and no size given, the pin holds Pro's 1920x1080, the lowest size DLSS takes.
constexpr Step DLSS_PRO_STEP = {1920, 1080};

// The code written at PATCH_VA for step `index`:
//   mov eax, index ; mov ecx, [rip -> count] ; dec ecx ; cmp eax, ecx ; cmovg eax, ecx ; jmp STORE_VA
inline std::array<uint8_t, ORIGINAL.size()> BuildPatch(uint32_t index) {
	std::array<uint8_t, ORIGINAL.size()> code {};
	auto put32 = [&](size_t at, uint32_t v) { std::memcpy(code.data() + at, &v, 4); };
	code[0] = 0xb8; // mov eax, imm32
	put32(1, index);
	code[5] = 0x8b; // mov ecx, [rip + disp32]
	code[6] = 0x0d;
	put32(7, static_cast<uint32_t>(COUNT_VA - (PATCH_VA + 11)));
	code[11] = 0xff; // dec ecx
	code[12] = 0xc9;
	code[13] = 0x39; // cmp eax, ecx
	code[14] = 0xc8;
	code[15] = 0x0f; // cmovg eax, ecx
	code[16] = 0x4f;
	code[17] = 0xc1;
	code[18] = 0xe9; // jmp rel32
	put32(19, static_cast<uint32_t>(STORE_VA - (PATCH_VA + 23)));
	return code;
}

// The game's active step: its index and the size its own table gives for it.
struct ActiveStep {
	uint32_t index;
	Step     size;
};

// Nothing when the values cannot be a step, as before the game has filled its table at boot.
constexpr std::optional<ActiveStep> MakeActiveStep(uint32_t active, uint32_t count, Step size) {
	if (count == 0 || count > 16 || active >= count || size.width == 0 || size.height == 0) {
		return std::nullopt;
	}
	return ActiveStep {active, size};
}

// True when the main program is the 1.05 eboot, the one whose addresses are known.
bool CanReadActiveStep();

// What the game's controller last stored for its render buffers, read live from guest memory.
std::optional<ActiveStep> ReadActiveStep();

} // namespace WolverineRenderStep

// Pins the render step when Config::GetRenderWidth/Height name a step of the mode's table (or, in Pro mode with a DLSS
// folder and no size given, at DLSS_PRO_STEP) and the main program is the 1.05 eboot; leaves any other title or build
// untouched. Call after relocation, before guest code runs.
void InstallWolverineRenderStep(Program* main_program);

} // namespace Loader

#endif // KYTY_LOADER_WOLVERINE_RENDER_STEP_H_
