#ifndef EMULATOR_SRC_KERNEL_SCANSTAMP_H_
#define EMULATOR_SRC_KERNEL_SCANSTAMP_H_

#include <cstdint>

// gt7-drawmemo-1118, KYTY_SCAN_STAMP (off by default): write stamps of guest memory, so that the
// GPU thread can take a draw-prep scanner result (drawPrep.h, ReadSet) without comparing the bytes
// the scanner read again.
//
// A global epoch grows with every stamp; a page's stamp (hashed, direct-mapped: pages that share a
// slot only make a check stricter) is the epoch of the last event that could change what a read
// of the page gives without a write fault:
//   - a host protection change of a tracked page (PageManager): stamped before a page becomes
//     writable, and after it becomes write-protected (the writes before it happened before);
//   - the emulator's writes into guest memory (JournalWrite: backing writes, downloads, stores in
//     place, WRITE_DATA, DMA fills and copies, file reads), stamped after the write;
//   - bytes marked GPU-written (BufferCache::MarkGpuWritten), stamped after the mark.
// And a layout epoch, moved before every change of the guest's mappings or of their host
// protection made outside the page tracker (map, unmap, commit, guest mprotect, live stacks,
// zeroed backing): a result prepared before one is checked byte by byte.
//
// A game thread's store to a page that is not write-protected leaves no stamp (a CPU-dirty page,
// a hot or "kept" page of the native memory branch, a live stack, private memory): the scanner
// marks the ranges it read on such pages, and the GPU thread compares their bytes.
//
//   KYTY_SCAN_STAMP=1  stamps are kept and used.
//   KYTY_SCAN_STAMP=2  verifier: the stamp's verdict is made and the bytes are compared too; a
//                      verdict that accepts what the bytes reject is counted ("falsos": 0).
namespace Libs::ScanStamp {

// 0 (off), 1 (on) or 2 (verifier), from KYTY_SCAN_STAMP (logged once).
[[nodiscard]] int  Mode();
[[nodiscard]] bool On();

// Any thread. The epoch now: every later stamp is larger.
[[nodiscard]] uint64_t Epoch();
// Any thread: [vaddr, vaddr + size) may read differently from now on.
void Note(uint64_t vaddr, uint64_t size);
// Whether a page of [vaddr, vaddr + size) was stamped after `epoch`.
[[nodiscard]] bool WrittenSince(uint64_t vaddr, uint64_t size, uint64_t epoch);

// The guest's mappings or their host protection are about to change outside the page tracker.
void LayoutChanging();
[[nodiscard]] uint64_t LayoutEpoch();

// Live thread stacks (GuestAddressSpace::AddLiveStack): the tracker's protections leave them
// writable. Registered before the layout moves; a range over one is never taken as watched.
void AddLiveStack(uint64_t start, uint64_t end);
void RemoveLiveStack(uint64_t start, uint64_t end);
[[nodiscard]] bool OverlapsLiveStack(uint64_t vaddr, uint64_t size);

} // namespace Libs::ScanStamp

#endif // EMULATOR_SRC_KERNEL_SCANSTAMP_H_
