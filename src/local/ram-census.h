#pragma once
// RAM census (KYTY_RAM_CENSUS=0 turns it off; on by default; Windows only): every 30 s one `[ram-census]` line (MiB)
// that splits the PC's RAM in use, read-only, from the game itself (no external tool):
//
//   sys    total / in use (total - available) / available
//   lists  modified page list (dirty pages waiting for the page file: "in use"), standby (normal + reserve + core),
//          free + zero
//   kern   nonpaged pool, paged pool resident, system driver resident, system code resident, system cache
//   self   working set / private working set / private commit / page faults per second
//   others the working sets of all other processes, and the 3 largest by name
//   vk     per Vulkan memory heap (D: device-local): usage / budget (VK_EXT_memory_budget, through VMA) and VMA's own
//          block bytes; host = the non-device-local heaps' usage (system RAM the driver holds for the emulator)
//   lean   KYTY_LEAN_HOSTMEM's skipped backing writes (lean-hostmem.h)
//   unexplained = in use - (self WS + others' WS + modified + nonpaged + paged resident + driver resident + code
//          resident): mostly driver-locked pages (GPU system memory) and other memory no working set holds.
//
// System counters come from PDH (pdh.dll loaded at run time, English counter paths: any Windows language). A thread
// at below-normal priority; it shares no lock with the game (the Vulkan heaps through a function vma.cpp registers).

#include <cstdint>

namespace RamCensus {

struct Heap {
	uint64_t usage  = 0; // the driver's figure (VK_EXT_memory_budget), bytes
	uint64_t budget = 0;
	uint64_t blocks = 0; // VMA's own memory blocks in the heap
	bool     device_local = false;
};

// Fills up to `max` heaps and returns how many (0: none known yet). Called on the census thread only.
using HeapQuery = int (*)(Heap* heaps, int max);
void SetHeapQuery(HeapQuery query); // (nullptr: none; waits for a query running meanwhile)

void Start(); // once; nothing when KYTY_RAM_CENSUS=0 or not Windows

} // namespace RamCensus
