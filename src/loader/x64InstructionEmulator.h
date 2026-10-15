#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <Zydis/DecoderTypes.h>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

// Used by the red-zone patcher ported from the Wolverine fork; reciprocal square root emulation is
// not ported, so these find and patch nothing.
[[nodiscard]] bool IsReciprocalSquareRoot(const ZydisDecodedInstruction& instruction,
                                         const ZydisDecodedOperand* operands);
uint64_t           PatchReciprocalSquareRoots(uint64_t address, uint64_t size);

[[nodiscard]] bool TryEmulate(void* native_context);

// A guest debug break, int 0x41 (the game's assertion handler breaks into an attached debugger with
// it): in user mode the CPU raises a general protection fault. Without a debugger it does nothing and
// the guest code after it runs (the game reports the failure and goes on). True when it was one.
[[nodiscard]] bool TrySkipDebugBreak(void* native_context);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
