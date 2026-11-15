#include "loader/wolverineBvhMagic.h"

#include "common/logging/log.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"
#include "loader/x64InstructionEmulator.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Loader {

namespace {

// movabs rdx, "PSR_BVHL" (0x3ee16a2); cmp [rcx], rdx (0x3ee16ac, 48 39 11).
constexpr uint64_t                 CHECK_VA = 0x3ee16a2;
constexpr std::array<uint8_t, 13>  ORIGINAL {0x48, 0xba, 0x50, 0x53, 0x52, 0x5f, 0x42, 0x56, 0x48, 0x4c, 0x48, 0x39, 0x11};
constexpr uint64_t                 CMP_OFFSET = 10;
constexpr size_t                   CMP_LENGTH = 3;
constexpr uint8_t                  RCX = 1;
constexpr uint8_t                  RDX = 2;
std::atomic<uint64_t>              g_cmp_rip = 0;

} // namespace

void InstallWolverineBvhMagic(Program* main_program) {
	std::string title_id;
	if (main_program == nullptr || !SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id != "PPSA03671") {
		return;
	}
	const char* value = std::getenv("KYTY_WOLVERINE_BVH_MAGIC");
	if (value != nullptr && value[0] == '0') {
		return;
	}
	const auto at = main_program->base_vaddr + CHECK_VA;
	if (std::memcmp(reinterpret_cast<const void*>(at), ORIGINAL.data(), ORIGINAL.size()) != 0) {
		Log::WriteToConsoleAndLog("Wolverine BVH magic: the eboot is not 1.05's; read back as before\n");
		return;
	}
	g_cmp_rip.store(at + CMP_OFFSET, std::memory_order_relaxed);
	Log::WriteToConsoleAndLog("Wolverine BVH magic: the check reads the backing\n");
}

bool WolverineBvhMagicCompare(void* native_context, uint64_t rip) {
	const auto at = g_cmp_rip.load(std::memory_order_relaxed);
	if (at == 0 || rip != at) {
		return false;
	}
	const uint64_t address = X64InstructionEmulator::SavedGpr(native_context, RCX);
	uint64_t       header  = 0;
	// Only a backing that already holds the magic is answered here. A new BVH's header reaches the backing only through
	// the read back (the backing holds 0xbaadf00d before it: 6 of ~97,000 checks at Trask); once there, it never changed.
	if (!Libs::LibKernel::Memory::TryReadBacking(address, &header, sizeof(header)) ||
	    header != X64InstructionEmulator::SavedGpr(native_context, RDX)) {
		return false;
	}
	X64InstructionEmulator::CompleteCompare(native_context, CMP_LENGTH, header,
	                                        X64InstructionEmulator::SavedGpr(native_context, RDX));
	return true;
}

} // namespace Loader
