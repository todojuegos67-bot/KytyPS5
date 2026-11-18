#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVEMEM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVEMEM_H_

#include <cstddef>
#include <cstdint>

// KYTY_NATIVE_MEM=1 (off by default): GT7's per-frame GPU pool as an upload heap, after the
// Demon's Souls port (ds-native-mem, KYTY_NATIVE_MEM=3), without write-protecting it page by page.
//
// What the game does (Ghidra, eboot 0x5710 / 0xacf0): a lock-free bump allocator hands out ranges
// of 2 MiB chunks, each from sceKernelAllocateDirectMemory(len 0x200000, align 0x200000, type 11 or
// 12) mapped with sceKernelMapDirectMemory(fixed address from a bump pointer, len 0x200000,
// prot 0xf2 (CPU write, GPU read/write), flags 0x90, align 0x200000): 0x10_0000_0000 upwards. The
// job threads write their PM4 chunks there (0x2aac0 closes a chunk: NOP / INDIRECT_BUFFER chain
// packets and a free-list node in its tail) and the draws' constants (0x45690: 4x4 matrices of a
// per-view cache, stride 0x1240; vmovups stores at 0x2e7fd7c / 0x2e8456d / 0x4bb93); the GPU reads
// it directly (unified memory). Nearly every write fault of a race lands there (the
// KYTY_FAULT_PROFILE "last" addresses: 0x10_0000_0000..0x10_0220_0000).
//
// In Kyty every upload of those pages (a binding's sync, the BDA walk, ~42 a frame) protects them
// again, and the job threads fault on them again: ~1750 faults a frame on ~1000 pages, ~18 us
// each (0.6-0.8 s a second in handlers, 19 % of Job#0).
//
// With it on: a write fault in the pool releases its whole 64 KB block at once (no second fault
// within 100 ms needed) and the page that faulted becomes "kept": writable, never protected, and
// made CPU-dirty again at every point after which the game's writes reach GPU work (a slice of a
// submission, a passed WAIT_REG_MEM), so every use after such a point uploads it again (the hot
// pages' contract, RegionManager::MarkHot), but only when its bytes changed since its last upload
// (64-bit hash, checked at those points), and once a frame whatever its hash. A kept page whose
// bytes did not change for KYTY_NATIVE_MEM_IDLE_US is tracked again as before; at most
// KYTY_NATIVE_MEM_CAP pages are kept. Nothing is freed, moved or mapped differently.
namespace Libs::Graphics::NativeMem {

// KYTY_NATIVE_MEM=1 (read once, logged).
[[nodiscard]] bool Enabled();
// KYTY_NATIVE_MEM_IDLE_US (default 3000, 100..1000000): how long a kept page's bytes stay
// unchanged before it is tracked again. KYTY_NATIVE_MEM_CAP (default 1024 pages, 4 MB): most
// pages kept at once (each costs a hash at every refresh that finds it uploaded).
[[nodiscard]] uint64_t IdleMicros();
[[nodiscard]] size_t   PageCap();

// sceKernelMapDirectMemory: whether a mapping has the pool chunks' signature (see above).
[[nodiscard]] bool IsPoolMapping(uint64_t vaddr, uint64_t len, int flags, uint64_t alignment,
                                 bool gpu_read_write);
// Pool chunks mapped and unmapped (any range: the chunks of it are forgotten).
void AddPool(uint64_t vaddr, uint64_t len);
void RemovePool(uint64_t vaddr, uint64_t len);
// Lock-free: whether vaddr lies in a pool chunk.
[[nodiscard]] bool InPool(uint64_t vaddr) noexcept;

// Counters for the line every 10 s ("Native mem:").
// A write fault in the pool (fault handler): whether it released its block, how many pages it
// made kept.
void NoteFault(bool released_block, size_t kept) noexcept;
// A refresh (GPU thread): at a slice or a wait, kept pages now, made CPU-dirty again, found
// written without a fault since their last upload, tracked again, hashed.
struct Refresh {
	size_t   kept      = 0;
	size_t   redirtied = 0;
	size_t   written   = 0;
	size_t   dropped   = 0;
	size_t   hashed    = 0;
	bool     at_wait   = false;
	bool     skipped   = false; // a wait the CPU did not release: no refresh
	uint64_t hash_ns   = 0;
};
void NoteRefresh(const Refresh& refresh) noexcept;
// GPU thread, after a refresh: the line every 10 s.
void MaybeReport(uint64_t now_ms);

} // namespace Libs::Graphics::NativeMem

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVEMEM_H_
