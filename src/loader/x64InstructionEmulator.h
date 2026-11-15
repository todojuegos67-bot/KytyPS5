#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstddef>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);
// Completes a faulting load into EAX as a zero read and resumes after it.
void SkipLoad(void* native_context, size_t length);
// Completes a faulting load into EAX with value and resumes after it.
void CompleteLoad(void* native_context, size_t length, uint32_t value);
// A general-purpose register of the saved context (0 rax ... 15 r15).
[[nodiscard]] uint64_t SavedGpr(void* native_context, uint8_t index);

// The arithmetic flags (CF PF AF ZF SF OF) a 64-bit `cmp left, right` leaves.
[[nodiscard]] constexpr uint64_t CompareFlags(uint64_t left, uint64_t right) {
	const uint64_t result = left - right;
	uint64_t       parity = result & 0xffu;
	parity ^= parity >> 4u;
	parity ^= parity >> 2u;
	parity ^= parity >> 1u;
	return (left < right ? 0x1u : 0u) | ((parity & 1u) == 0 ? 0x4u : 0u) | ((left ^ right ^ result) & 0x10u) |
	       (result == 0 ? 0x40u : 0u) | ((result >> 63u) != 0 ? 0x80u : 0u) |
	       ((((left ^ right) & (left ^ result)) >> 63u) != 0 ? 0x800u : 0u);
}
static_assert(CompareFlags(5, 5) == 0x44u);
static_assert(CompareFlags(0, 1) == 0x95u);
static_assert(CompareFlags(0x8000000000000000ull, 1) == 0x814u);
static_assert(CompareFlags(0x10, 0x1) == 0x14u);
// Completes a faulting `cmp [mem], reg` (64-bit) with the memory operand's value and resumes after it.
void CompleteCompare(void* native_context, size_t length, uint64_t memory, uint64_t reg);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
