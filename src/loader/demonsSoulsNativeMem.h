#pragma once
// KYTY_NATIVE_MEM (off by default): what a native PC port of Demon's Souls would not have in its memory path, measured
// and (level 2) the first part of it removed. Static notes: /home/claude/static/native-port-plan.md.
//
// KYTY_NATIVE_MEM=1  census only, nothing in the emulator's behaviour changes:
//   - at load, the engine code this relies on is checked byte for byte (01.007.000; any mismatch: off, with the reason);
//   - at sceVideoOutRegisterBuffers2 (CRenderDisplay ctor, right after RenderTargetManager::InitSharedMemory) the two
//     render-target arenas are read from the engine's own objects and checked (vtable, its relocated Alloc/Free slots,
//     name, EMemoryType, base and size): RTSharedAllocator and RTPersistAllocator, GPU-only on a native port;
//   - every write-tracking fault (window / write / read), tracker protection call and GPU write-back into guest
//     memory is attributed to the engine allocator whose page run holds the address (the engine's allocator registry,
//     walked every 10 s), and printed every 10 s ("[native-mem]" lines); every 30 s the resident RAM of each allocator's
//     runs is sampled (one page per 64 KiB, the game's view and the backing alias).
// KYTY_NATIVE_MEM=2  the census, and the render-target arenas kept GPU-only as on a native port: an image that lies
//   wholly inside an arena is never downloaded into guest RAM by the image collector or the linear-image readback (it
//   stays in video memory, as a tiled render target already does); while the GPU cannot make images (the collector's
//   "failing" state) the old behaviour comes back. Guest reads of such memory still fault and read the GPU's bytes back.
// KYTY_NATIVE_MEM=3  level 2, and (after more byte checks; any mismatch: level 2):
//   - per-frame CPU->GPU arenas as a native port's upload heap (graphics/host_gpu/uploadHeap.h): the page runs of the
//     GPU "Default TaggedArena" (parent of GameToGPU / RenderSingleFrameGPU) and of the "RenderContextGPU" memory
//     context's allocators (render-context command and descriptor memory) are marked; the buffer cache never
//     write-protects their pages and never makes them clean: every GPU use copies what it reads (once per observation
//     epoch of the GPU thread). No write fault, no unprotect/reprotect for them, no write missed;
//   - streamed mips: the memcpy of CTexture stream-in (0x901459, from ProcessTextureDataInternal 0xaa4a60) calls a host
//     copy that tells the trackers once for the whole mip (as a file read does) instead of faulting page by page.
// Nothing here frees, decommits, trims or moves memory. Level 3 rewrites one guest call (0x901459) to a cave.

#include <atomic>
#include <cstdint>

namespace Loader {
struct Program;
namespace DemonsSoulsNativeMem {

enum Event : uint32_t {
	FaultWindow = 0, // CPU write fault served by the write window
	FaultWrite  = 1, // other CPU write fault on a watched page
	FaultRead   = 2, // CPU read fault on a page the GPU wrote
	Protect     = 3, // a tracker protection change (VirtualProtect / mprotect)
	WriteBack   = 4, // GPU bytes written into guest memory (readback, download)
	EventKinds  = 5,
};

// 0 off, 1 census, 2 census + GPU-only render-target arenas.
extern std::atomic<int> g_level;
[[nodiscard]] inline bool On() noexcept { return g_level.load(std::memory_order_relaxed) > 0; }
// Lock-free; any thread.
void Record(Event kind, uint64_t vaddr, uint64_t size) noexcept;
inline void Note(Event kind, uint64_t vaddr, uint64_t size) noexcept {
	if (On()) Record(kind, vaddr, size);
}

// Whether [vaddr, vaddr + size) lies wholly inside one render-target arena (false until they were found).
[[nodiscard]] bool InRtArena(uint64_t vaddr, uint64_t size) noexcept;
// Level 2: an image over such memory is kept in video memory instead of being downloaded into guest RAM. `why`:
// 0 the image collector, 1 the linear-image readback. Counts what it kept.
[[nodiscard]] bool KeepGpuOnly(uint64_t vaddr, uint64_t size, int why) noexcept;

// Load time (the runtime linker's patch list): the static checks.
void Install(Program* program);
// sceVideoOutRegisterBuffers2: the arenas, then the census thread. Once.
void OnRegisterBuffers() noexcept;

} // namespace DemonsSoulsNativeMem
} // namespace Loader
