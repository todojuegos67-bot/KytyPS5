#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <Zydis/DecoderTypes.h>
#include <cstdint>
#include <string>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool IsReciprocalSquareRoot(const ZydisDecodedInstruction& instruction,
                                         const ZydisDecodedOperand* operands);
uint64_t           PatchReciprocalSquareRoots(uint64_t address, uint64_t size);
[[nodiscard]] bool TryEmulate(void* native_context);
// Instructions emulated through TryEmulate (each one a host exception, several microseconds; --amd-cpu's VRSQRTPS
// sites that could not be relocated among them), counted since start, and the busiest guest addresses since the
// last call ("rip=0x... n=..." pairs, then cleared). For the [traps] line of the frame stats (videoOut.cpp).
[[nodiscard]] uint64_t    TrapCount() noexcept;
[[nodiscard]] std::string TakeTopTrapSites(int count);

// Reads [vaddr, vaddr + size) for a load that faulted at fault_vaddr; false to decline.
using LoadReader = bool (*)(uint64_t fault_vaddr, uint64_t vaddr, void* data, uint64_t size);
// Completes a faulting load whose memory operand covers fault_vaddr by reading the operand through
// `read` instead of the faulting mapping, then steps past the instruction. Pure loads only: MOV,
// MOVZX, MOVSX, MOVSXD, CMP and TEST with a memory operand, and SSE/AVX moves from memory into a
// vector register. False, with the context untouched, for anything else or when `read` declines.
[[nodiscard]] bool TryEmulateLoad(void* native_context, uint64_t fault_vaddr, LoadReader read);
// Diagnostics: the mnemonic of the instruction at the context's RIP.
[[nodiscard]] const char* DescribeInstruction(void* native_context);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
