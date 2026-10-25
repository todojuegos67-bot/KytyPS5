#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstdint>
#include <string>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);

// astro-perf-1025: instructions emulated in the illegal-instruction handler (each one a host exception, several
// microseconds), counted since start, and the busiest guest addresses since the last call ("rip=0x... n=..."
// pairs, then cleared). For the [traps] line of the frame log (local/metrics.cpp).
[[nodiscard]] uint64_t TrapCount() noexcept;
[[nodiscard]] std::string TakeTopTrapSites(int count);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
