#ifndef EMULATOR_SRC_LOADER_WOLVERINEBVHMAGIC_H_
#define EMULATOR_SRC_LOADER_WOLVERINEBVHMAGIC_H_

#include <cstdint>

namespace Loader {

struct Program;

// Marvel's Wolverine (PPSA03671) 1.05: the game's BVH magic check, `cmp [rcx], rdx` with rdx = "PSR_BVHL" (eboot
// 0x3ee16ac), reads a header in the geometry pool that the CPU writes and no shader does. When the Pro refits write
// nodes on the same page, the check faults and reads the whole page back (a GPU drain) for 8 bytes the GPU never
// changed. On (KYTY_WOLVERINE_BVH_MAGIC unset or not 0), that fault compares against the backing instead and the page
// stays tracked. Nothing on another title or version.
void InstallWolverineBvhMagic(Program* main_program);

// The fault handler, on a read fault: true when it was that instruction and it is now completed from the backing.
[[nodiscard]] bool WolverineBvhMagicCompare(void* native_context, uint64_t rip);

} // namespace Loader

#endif /* EMULATOR_SRC_LOADER_WOLVERINEBVHMAGIC_H_ */
