#ifndef EMULATOR_SRC_LOCAL_SOFT_RELEASE_H_
#define EMULATOR_SRC_LOCAL_SOFT_RELEASE_H_
// KYTY_SOFT_RELEASE=1 (default off): RAM the emulator has freed but still holds is given back to Windows without the
// page file: never EmptyWorkingSet, never SetProcessWorkingSetSize(Ex) (the old trim, KYTY_TRIM_RAM_SECONDS, forced
// page-outs and page-ins and bug-checked Windows, 0x1A MEMORY_MANAGEMENT; with KYTY_SOFT_RELEASE=1 it stays off).
// A background thread, every KYTY_SOFT_RELEASE_SECONDS (at least and by default 30), when the last 2 s of frames were
// all within budget (KYTY_SOFT_RELEASE_BUDGET_MS, default 20; a busy or fighting frame skips the pass):
//   - the process's heaps compacted (HeapCompact, _heapmin): their free blocks decommitted, nothing written to disk;
//   - each registered source's free memory dropped (MEM_RESET / DiscardVirtualMemory / MEM_DECOMMIT, its own choice).
// It runs at low memory priority (its own thread only; the process's priority is left alone). Each pass logs the
// MiB released (private bytes before less after) and the total. The guest's memory is never touched.
#include <cstdint>

namespace SoftRelease {

[[nodiscard]] bool Enabled();
// Starts the thread once (nothing when off or not on Windows).
void Start();
// A source of freed memory the emulator keeps (a pool's free list, a scratch buffer while idle): `release` runs on the
// soft-release thread, must drop only memory that holds no live data (discard or decommit, never the page file) and
// returns the bytes it dropped. Not for guest memory.
void Register(const char* name, uint64_t (*release)());
// Windows: the calling thread's memory priority low (MEMORY_PRIORITY_LOW): the pages it touches are the first Windows
// takes under real memory pressure; nothing is forced. For background and cold-cache threads only.
void LowerThreadMemoryPriority();
// Helpers for sources: drop the pages of [address, address + size) whose contents are no longer needed (they stay
// committed; the next write gets zero or old bytes). Page-aligned inside; returns the bytes covered.
uint64_t DiscardPages(void* address, uint64_t size);

} // namespace SoftRelease

#endif // EMULATOR_SRC_LOCAL_SOFT_RELEASE_H_
