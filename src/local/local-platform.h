#ifndef EMULATOR_SRC_LOCAL_LOCAL_PLATFORM_H_
#define EMULATOR_SRC_LOCAL_LOCAL_PLATFORM_H_
// The few host-platform helpers the todojuegos67-bot features need on this tree (frame generation's
// Streamline check, the RAM guard): a subset of that fork's src/local/local-platform.h.
#include <cstdint>
#include <string>

namespace LocalPlatform {

// The file is signed by NVIDIA Corporation (Authenticode, the leaf certificate of its first signature).
bool IsNvidiaSigned(const std::wstring& path);

struct MemoryStatus {
	uint64_t total_phys = 0, avail_phys = 0;       // RAM
	uint64_t commit_limit = 0, commit_total = 0;    // system commit charge
	uint64_t private_bytes = 0, working_set = 0;    // this process
	bool     valid = false;
};
MemoryStatus QueryMemory();
// Windows' low-memory resource notification is signaled (always false elsewhere).
bool LowMemorySignaled();
// Heaps compacted (HeapCompact on the process heaps): bytes the call reported as the largest free block.
uint64_t CompactHeaps();

} // namespace LocalPlatform

#endif // EMULATOR_SRC_LOCAL_LOCAL_PLATFORM_H_
