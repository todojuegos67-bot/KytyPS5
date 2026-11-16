#include "loader/wolverinePhysicsStep.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace Loader {

using namespace WolverinePhysicsStep;

void InstallWolverinePhysicsStep(Program* main_program) {
	std::string title_id;
	if (main_program == nullptr || !SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id != "PPSA03671") {
		return;
	}
	const char* value = std::getenv("KYTY_WOLVERINE_PHYSICS_STEP");
	if (value != nullptr && value[0] == '0') {
		Log::WriteToConsoleAndLog("Wolverine physics step: off (the game's fixed step)\n");
		return;
	}
	const auto base = main_program->base_vaddr;
	auto*      code = reinterpret_cast<uint8_t*>(base + PATCH_VA);
	if (std::memcmp(code, ORIGINAL.data(), ORIGINAL.size()) != 0) {
		Log::WriteToConsoleAndLog("Wolverine physics step: the eboot is not 1.05's; fixed step\n");
		return;
	}
	if (main_program->title_hook_vaddr == 0) {
		Log::WriteToConsoleAndLog("Wolverine physics step: no title hook area (needs --redzone); fixed step\n");
		return;
	}
	auto* hook = reinterpret_cast<uint8_t*>(main_program->title_hook_vaddr);
	std::memset(hook, 0, 16); // the state: last stored phys, manager's phys
	Xbyak::CodeGenerator gen(TITLE_HOOK_SIZE - 16, hook + 16);
	EmitPhysicsStep(gen, {base + PHYS_VA, base + OVR_VA, base + MIN_VA, main_program->title_hook_vaddr},
	                code + ORIGINAL.size());
	EXIT_IF(Xbyak::GetError() != 0);
	EXIT_IF(!Common::VirtualMemory::FlushInstructionCache(main_program->title_hook_vaddr, 16 + gen.getSize()));

	// jmp rel32 to the hook; the rest of the replaced bytes is never reached (the hook resumes at RESUME_VA).
	std::array<uint8_t, ORIGINAL.size()> patch {};
	patch.fill(0xcc);
	const auto rel = static_cast<int32_t>(reinterpret_cast<int64_t>(hook + 16) - static_cast<int64_t>(base + PATCH_VA + 5));
	EXIT_IF(static_cast<int64_t>(rel) != reinterpret_cast<int64_t>(hook + 16) - static_cast<int64_t>(base + PATCH_VA + 5));
	patch[0] = 0xe9;
	std::memcpy(patch.data() + 1, &rel, 4);
	std::memcpy(code, patch.data(), patch.size());
	EXIT_IF(!Common::VirtualMemory::FlushInstructionCache(base + PATCH_VA, patch.size()));
	Log::WriteToConsoleAndLog("Wolverine physics step: Havok steps by the frame time below the target rate\n");
}

} // namespace Loader
