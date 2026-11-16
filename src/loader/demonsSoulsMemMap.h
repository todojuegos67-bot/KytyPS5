#pragma once

#include <cstddef>
#include <cstdint>

// Demon's Souls 01.007.000 (eboot at 0x900000000): a map of what the game keeps where, for the [faults] report only
// (measurement: nothing here changes the game or the emulator's behaviour). KYTY_MEMMAP=0 turns it off (default on).
//
// Labels (each [faults] row gets "label N%" = the share of its 2 MiB, in 4 KiB pages, that the label covers):
//   dcb      command buffers submitted to the GPU in the last 5 s (AgcDriverSubmit* -> NoteSubmit)
//   ctx      the game's CRenderContext/CComputeContext DCB blocks (read from CRenderBackend, last 5 s) and the 7
//            small backend DCBs
//   cpuheap  HeapAllocator of CPUMemory (type 1)   gpuheap  HeapAllocator of GPUMemory (type 2)
//   pool     the backend's 112 MiB block           ring     RenderContextFrameCPU 130 MiB + FrameGPU 20 MiB
//   others   every other allocator the engine registered (table eboot+0x2ea0df0), by its name: cpupage/gpupage,
//            cpubkt/gpubkt, RTShared..., printed when >= 1 %
// A background thread reads the game's memory (GuardedCopy only) every 100 ms (contexts) and 2 s (allocators) and
// prints "[memmap]" lines when the set changes (at most every 10 s).
namespace Loader::DemonsSoulsMemMap {

// Any thread (the Agc submit HLE): a command buffer the GPU is given. Lock-free.
void NoteSubmit(const void* commands, uint64_t size_in_bytes) noexcept;

// The tags for [begin, end) into `out` ("dcb N%  ctx N%  ..."); empty when off. Lock-free, no allocation, no guest
// memory reads: safe in the fault handler.
void Tags(uint64_t begin, uint64_t end, char* out, size_t out_size) noexcept;

// KYTY_VOLATILE=1 (and the map on): the 4 MiB granules (address >> 22) of memory found to be rewritten every frame,
// append-only and deduplicated, each entry granule << 1 | kind (0 = an allocator named by KYTY_VOLATILE_ALLOCATORS,
// default "DefaultTagg", plus KYTY_VOLATILE_EXTRA_ALLOCATORS; 1 = ctx: the render/compute context DCB blocks and the
// small backend DCBs). Copies entries [from, from + max) into `out`; returns how many. Lock-free, any thread.
size_t VolatileGranules(size_t from, uint64_t* out, size_t max) noexcept;
// KYTY_BIG_WINDOW=1: whether vaddr is in a found DefaultTagg/ctx granule (lock-free; fault handler safe).
bool InBigWindowGranule(uint64_t vaddr) noexcept;
// (v4, KYTY_VOLATILE) The command buffers submitted (NoteSubmit) since *cursor, as [begin, end) pairs into out[2 * i],
// at most `max`; *cursor moves past them (one reader: the GPU thread). Lock-free.
size_t SubmitsSince(uint64_t* cursor, uint64_t* out, size_t max) noexcept;
// (v4, KYTY_VOLATILE) The context DCB block (or the small DCBs' block) holding `address`, from the last poll (100 ms).
bool CtxBlockOf(uint64_t address, uint64_t* begin, uint64_t* end) noexcept;

// (ram1) Which of the game's render-target arenas holds all of [begin, end): 1 RTSharedAllocator (687 MiB, targets
// aliased inside the frame), 2 RTPersistAllocator (897 MiB, targets kept across frames), 0 none or not read yet. Read
// from the allocator table each 2 s like the other labels. Lock-free, no allocation: any thread.
int RtArenaOf(uint64_t begin, uint64_t end) noexcept;

// (ram1; ram2: off unless KYTY_MEMMAP_RESIDENCY=1) Every 10 s (Windows; needs the map): "[memmap-ram]" lines with, per
// label, the MiB the game has mapped there and the MiB in RAM (working set, sampled every 64 KiB or more through the
// game's view and the emulator's alias), plus the RAM of the label's pages the game no longer maps (unmapped but still
// resident: freed memory the emulator keeps) and the system's free / standby / modified lists when Windows tells.
// Measurement only, on the map's own thread.

// (ram2) The game's page allocator's own free memory (01.007.000, the code checked; false otherwise or before it
// starts): `cached_bytes` in released blocks it keeps mapped (invisible to the emulator: no unmap), `unmapped_bytes`
// free in its bitmap (not mapped). Guarded reads, no lock: any thread. Works with the map off too.
bool PageAllocatorFree(uint64_t* cached_bytes, uint64_t* unmapped_bytes) noexcept;

// (ram3, KYTY_DS_FREECACHE_RESET) The free blocks of the game's page allocator (01.007.000, the code checked): the
// blocks ReleasePages 0x80f550 keeps in each memory category's free list (mapped or partly mapped, their contents
// dead: the first-fit reuse 0x80e7a0 hands them out again as new memory). `fn` is called once, WITH THE GAME'S
// ALLOCATOR SPINLOCK HELD (eboot+0x2ee4b60, taken as the game takes it): no game thread can take a block out of the
// lists, map or unmap allocator memory until it returns, so it must be short (no waits, no emulator lock a game thread
// may hold while it allocates). Returns 0 done, 1 not this game/version or not started, 2 the lock was busy (try
// later), 3 the lists did not validate (nothing done). `cached_bytes`: the allocator's own count of bytes in free
// blocks. One caller at a time.
struct FreeBlock {
	uint64_t base = 0, size = 0, mapped = 0; // (mapped: the bytes the allocator counts as still mapped)
};
using FreeBlocksFn = void (*)(void* user, const FreeBlock* blocks, size_t count);
int WithPageAllocatorFreeBlocks(FreeBlocksFn fn, void* user, uint64_t* cached_bytes) noexcept;

// (audit1, KYTY_DS_WRITE_AUDIT) The ranges of the last snapshot (2 s) whose label is one of `labels` (the names the
// [faults] report uses: "ring", "ctx" - there only the small DCBs' block -, "DefaultTagg", "cpuheap"...), as
// [begin, end) pairs into out[2 * i], at most `max_pairs`. Returns how many; SIZE_MAX when there is no snapshot to
// read (the map is off, has not read the game yet, or is being replaced: ask again). Lock-free, no guest read.
size_t RangesOf(const char* const* labels, size_t label_count, uint64_t* out, size_t max_pairs) noexcept;
// (audit1) The label of the smallest range of the last snapshot that holds `address` ("ctx" for a render context's
// DCB block seen in the last 5 s when no range does) into `out`; false and "" when none does or there is no snapshot.
// Lock-free, no guest read; a scan of the whole snapshot: for reports, not for the fault path.
bool LabelAt(uint64_t address, char* out, size_t out_size) noexcept;

} // namespace Loader::DemonsSoulsMemMap
