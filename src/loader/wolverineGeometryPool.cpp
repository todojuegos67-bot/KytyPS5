#include "loader/wolverineGeometryPool.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Loader {

namespace {

// GeometryBufferManager's constructor (sub_3EEEDE0): mov word [rbx+1188h], 1. The word is two flags: +4488 small/large
// heaps (1) and +4489 lazy commit (0). With +4489 = 1, init only reserves the pool (sub_39772A0) and AllocateBuffer
// (sub_3EEFFE0) maps the 2 MiB pages under each block once ("CommitIABufferAllocator"); nothing unmaps them.
constexpr uint64_t                FLAGS_STORE_VA = 0x3eeeee9;
constexpr std::array<uint8_t, 9> ORIGINAL {0x66, 0xc7, 0x83, 0x88, 0x11, 0x00, 0x00, 0x01, 0x00};
constexpr std::array<uint8_t, 9> LAZY {0x66, 0xc7, 0x83, 0x88, 0x11, 0x00, 0x00, 0x01, 0x01};

} // namespace

void InstallWolverineGeometryPool(Program* main_program) {
	std::string title_id;
	if (main_program == nullptr || !SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id != "PPSA03671") {
		return;
	}
	const char* value = std::getenv("KYTY_SPARSE_GEOMETRY");
	if (value != nullptr && value[0] == '0') {
		return;
	}
	const auto base = main_program->base_vaddr;
	auto*      code = reinterpret_cast<uint8_t*>(base + FLAGS_STORE_VA);
	if (std::memcmp(code, ORIGINAL.data(), ORIGINAL.size()) != 0) {
		Log::WriteToConsoleAndLog("Wolverine geometry pool: the eboot is not 1.05's; mapped whole\n");
		return;
	}
	std::memcpy(code, LAZY.data(), LAZY.size());
	EXIT_IF(!Common::VirtualMemory::FlushInstructionCache(base + FLAGS_STORE_VA, LAZY.size()));
	Log::WriteToConsoleAndLog("Wolverine geometry pool: lazy commit (pages mapped as blocks land)\n");
}

} // namespace Loader
