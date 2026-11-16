#ifndef EMULATOR_SRC_LOCAL_RAIZ_WB_GUARD_H_
#define EMULATOR_SRC_LOCAL_RAIZ_WB_GUARD_H_

// (crash-ba2b1f, KYTY_RAIZ=1) A GPU write-back must not replace bytes the game wrote after the GPU took the page.
// 20261116-043751: a job worker's shared_ptr control blocks (heap 0x2ea0ce8, page 0x37b97e000) were replaced by the
// page's old contents (0x3c00000000000000) by the "download" write-back of 0x37b961000+0x1f000 - the game's writes on
// that page had reached no fault and no upload -, and the vtable call through them crashed (eboot+0xba2b1f).
//  - When pages become GPU-owned (RegionManager::UpdateGpuProtection) their write watch is reset; the emulator's own
//    writes (Memory::WriteBacking, its write windows, the kept bytes put in) reset it after them. A page whose bit is
//    set at a GPU write-back was written by the game while the GPU owned it: KYTY_RAIZ_WB_GUARD=1 (default with raiz)
//    leaves that page alone (the game's bytes win, as on the console where its write came later) and counts it;
//    =2 only counts and logs it (the write-back goes on as before); =0 off (no reset, no check).
//  - The kept bytes put in while the game's threads run (memoryAddressSpace.inc NativePendingMergeOpen): a game write
//    that lands in the open page wins (compare-and-swap per 8 bytes against the bytes the page had), and one that lands
//    before its final protection is found and the page marked here (and queued as CPU-written).
// Definitions: kernel/memory.cpp.

#include <cstdint>
#include <vector>

namespace Libs::LibKernel::Memory {

// 0 off, 1 guard (skip), 2 count only. Read once.
[[nodiscard]] int  RaizWbGuardMode() noexcept;
[[nodiscard]] inline bool RaizWbGuardOn() noexcept {
	return RaizWbGuardMode() != 0;
}
// [vaddr, vaddr + size) became GPU-owned: write watch reset, earlier marks forgotten.
void RaizWbGpuOwned(uint64_t vaddr, uint64_t size) noexcept;
// The emulator wrote [vaddr, vaddr + size): its write-watch bits are not the game's.
void RaizWbEmulatorWrote(uint64_t vaddr, uint64_t size) noexcept;
// A game write found where it should have faulted (`where`: the path that found it); the page is marked.
void RaizWbNoteGameWrote(uint64_t page, const char* where) noexcept;
// The pages of [vaddr, vaddr + size) the game wrote since they became GPU-owned (write watch or marks), sorted.
void RaizWbGameWritten(uint64_t vaddr, uint64_t size, std::vector<uint64_t>* pages) noexcept;
// A write-back's page left alone (mode 1) or written over the game's bytes (mode 2): counted, the first 32 printed.
void RaizWbNoteSkipped(uint64_t page, uint64_t address, uint64_t size, const char* source, bool skipped) noexcept;
// Kept bytes put in with a game write in the page meanwhile: `words` of 8 bytes the game's kept.
void RaizWbNotePutInRace(uint64_t page, uint64_t words, bool after) noexcept;
// Crash reports: the trackers' state of a page, read without locks (BufferCache registers it). Bits: RaizPeek*.
enum : uint32_t {
	RaizPeekRegion   = 1u << 0u,
	RaizPeekCpuDirty = 1u << 1u,
	RaizPeekGpuDirty = 1u << 2u,
	RaizPeekWritable = 1u << 3u,
	RaizPeekReadable = 1u << 4u,
	RaizPeekNoWatch  = 1u << 5u,
	RaizPeekNoSplit  = 1u << 6u,
};
using RaizPagePeek = bool (*)(void* user, uint64_t page, uint32_t* bits);
void RaizWbSetPagePeek(RaizPagePeek peek, void* user) noexcept;
// Crash reports: one line of a guest page's state (host protection, trackers, kept bytes, write watch, marks).
void PrintGuestPageState(uint64_t page) noexcept;

} // namespace Libs::LibKernel::Memory

#endif // EMULATOR_SRC_LOCAL_RAIZ_WB_GUARD_H_
