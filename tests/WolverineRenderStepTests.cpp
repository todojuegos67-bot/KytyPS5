#include "loader/wolverineRenderStep.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using namespace Loader::WolverineRenderStep;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WolverineRenderStepTests: failed: %s\n", message);
		std::abort();
	}
}

int32_t Rel32(const uint8_t* at) {
	int32_t v = 0;
	std::memcpy(&v, at, 4);
	return v;
}

void TestSteps() {
	Check(FindStep(1664, 936) == 8, "the lowest step is not the last");
	Check(FindStep(1920, 1080) == 6, "1920x1080 is not step 6");
	Check(FindStep(2560, 1440) == 3, "2560x1440 is not step 3");
	Check(FindStep(3328, 1872) == 0, "the highest step is not the first");
	Check(FindStep(1280, 720) == -1, "a size that is no step was found");
	Check(FindStep(0, 0) == -1, "auto was found as a step");
	// Pro mode's table: eleven steps, 1920x1080 at 8 (base's 1664x936 index), 1664x936 last.
	Check(FindStep(DLSS_PRO_STEP.width, DLSS_PRO_STEP.height, true) == 8, "Pro's 1920x1080 is not step 8");
	Check(FindStep(1664, 936, true) == 10 && FindStep(3840, 2160, true) == 0, "Pro's ends are not 10 and 0");
	Check(FindStep(3840, 2160) == -1, "base's table holds 3840x2160");
	Check(StepAt(true, 11).width == 0 && StepAt(false, 9).width == 0, "a step past a table's end");
}

void TestActiveStep() {
	const auto step = MakeActiveStep(8, 9, {1664, 936});
	Check(step.has_value() && step->index == 8 && step->size.width == 1664 && step->size.height == 936,
	      "a valid step was refused");
	Check(!MakeActiveStep(0, 0, {0, 0}).has_value(), "a table the game has not filled gave a step");
	Check(!MakeActiveStep(9, 9, {1664, 936}).has_value(), "an index past the count gave a step");
	Check(!MakeActiveStep(8, 9, {0, 936}).has_value(), "an empty table entry gave a step");
	Check(!MakeActiveStep(8, 17, {1664, 936}).has_value(), "an implausible count gave a step");
}

void TestPatch() {
	for (uint32_t index = 0; index < STEPS.size(); index++) {
		const auto code = BuildPatch(index);
		Check(code.size() == ORIGINAL.size(), "the patch does not cover the replaced instructions exactly");
		Check(code[0] == 0xb8 && Rel32(&code[1]) == static_cast<int32_t>(index), "mov eax, index");
		Check(code[5] == 0x8b && code[6] == 0x0d, "mov ecx, [rip + disp32]");
		Check(PATCH_VA + 11 + Rel32(&code[7]) == COUNT_VA, "the count load does not read 0xBEAB128");
		Check(code[11] == 0xff && code[12] == 0xc9, "dec ecx");
		Check(code[13] == 0x39 && code[14] == 0xc8, "cmp eax, ecx");
		Check(code[15] == 0x0f && code[16] == 0x4f && code[17] == 0xc1, "cmovg eax, ecx");
		Check(code[18] == 0xe9 && PATCH_VA + 23 + Rel32(&code[19]) == STORE_VA,
		      "the jump does not reach the controller's store");
	}
	// The store the jump reaches compares and writes the active step at 0x9B7BA54.
	Check(STORE[0] == 0x3b && STORE[1] == 0x05 && STORE_VA + 6 + Rel32(&STORE[2]) == ACTIVE_VA,
	      "the store does not compare with 0x9B7BA54");
	// The replaced bytes start with "cmp byte [0xBEAB728], 0", whose rip is 7 bytes on.
	Check(ORIGINAL[0] == 0x80 && ORIGINAL[1] == 0x3d && PATCH_VA + 7 + Rel32(&ORIGINAL[2]) == 0xBEAB728,
	      "the replaced test is not the fixed-step flag's");
}

} // namespace

int main() {
	TestSteps();
	TestActiveStep();
	TestPatch();
	std::printf("WolverineRenderStepTests: passed\n");
	return 0;
}
