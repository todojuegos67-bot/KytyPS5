#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVETEX_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVETEX_H_

// KYTY_NATIVE_TEX=1 (Windows; off by default, and off means no behavior change): texture bytes go to video memory as on
// a native PC port. The guest hooks (the game's texture loads: file reads into texture memory, the streamer's mip copies)
// hand the bytes here instead of storing them into guest memory; the guest pages are never written, so with
// KYTY_GUEST_LAZY_COMMIT they are never committed.
//
//  - OnTextureBytes: the bytes are copied ONCE into a host staging ring (allocated once at start-up: device-local and
//    host-visible with ReBAR, else host-visible; KYTY_NATIVE_TEX_RING_MB, default 256) and the guest range becomes
//    "GPU-authoritative": its guest pages hold nothing valid. The images over it are refreshed from the ring (the texture
//    cache's InitializeImage: one GPU detile per load, no per-frame copy); an image made later uploads from the ring when
//    it is first used. Once uploaded, the image holds the only copy and the ring space is reused.
//  - Every other reader gets correct bytes ("materialize": the bytes written into guest memory once, then normal
//    tracking): a CPU touch of the range (its pages are no-access while authoritative: the fault comes here), a host write
//    over part of it, an image over it that cannot be uploaded from the ring (format/layout the native path does not take,
//    a subresource only partly in the ring), the owning image going away (collector, unmap of a view), another image over
//    the bytes another image owns.
//  - A load's first and last partial pages are written into guest memory (they may share a page with other data);
//    only its whole pages become authoritative.
//  - Bounded memory: the ring is fixed. Bytes no image took stay in the ring until the ring passes 75% full; then the
//    oldest are written to guest memory (the old behavior for those textures), at most 16 MiB a flip, until it is below
//    50%. A load that finds no room is refused (the caller stores to guest memory, as without this). The range metadata
//    is capped (MaxEntries).
//
// Logged every 30 s: "NativeTex: ..." (MiB routed, ranges, uploads from the ring, fallbacks by reason, ring use).

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
struct GraphicContext;

namespace NativeTex {

// ---- Guest hooks (any thread) ----------------------------------------------------------------------------------------

// KYTY_NATIVE_TEX=1 on Windows (read once).
[[nodiscard]] bool Enabled() noexcept;

// The game would store `size` bytes from `src` (host-readable) at guest `guest_dst`. True: taken; the caller must NOT
// write guest memory. False: the caller stores them as usual (disabled, not ready yet, too small, ring full, or not
// texture memory B's VirtualMemoryGPU hooks declared: OnTextureCommit - the GPU-context heap's textures, whose frees
// are not hooked, stay in guest memory).
[[nodiscard]] bool OnTextureBytes(uint64_t guest_dst, const void* src, uint64_t size);

// The game would move `size` bytes from guest `src` to guest `dst` (memmove semantics, the source keeps its bytes).
// True: done here (ring bytes shared, or a GPU copy of the image's bytes into the ring); the caller must NOT touch guest
// memory. False: the caller moves guest memory as usual (any authoritative bytes of `src` were written into guest memory
// first, and those of `dst` dropped).
[[nodiscard]] bool OnTextureMove(uint64_t dst, uint64_t src, uint64_t size);

// The game released the texture memory [addr, addr + size): its authoritative bytes are dropped (nothing written back).
void OnTextureRelease(uint64_t addr, uint64_t size);

// The game committed [addr, addr + size) for texture data (B's VirtualMemoryGPU hook): until something writes it, its
// content is undefined, so an image over it uploads zeros there instead of reading (and committing) guest memory.
void OnTextureCommit(uint64_t addr, uint64_t size);
// Pages of a texture's run given back inside the run (a streamed-out mip, B's ResizePages op 2): the texture keeps the
// memory, its content is undefined again (as OnTextureCommit). Unlike a release, no other allocator gets these pages.
void OnTextureDecommit(uint64_t addr, uint64_t size);

// ---- Emulator side (the texture cache) -------------------------------------------------------------------------------

// A range whose bytes are not in guest memory.
struct Piece {
	uint64_t begin = 0, end = 0;
	bool     owned = false;  // the bytes are in the image `image` (serial `serial`), else in the ring at `ring_offset`
	bool     undefined = false; // (not owned) no bytes at all: committed texture memory nothing wrote yet (its content is
	                            // undefined: an upload may take zeros, a touch takes the guest's bytes as they are)
	uint64_t ring_offset = 0;
	uint64_t image = 0, serial = 0; // (owned: ImageId as (generation << 32) | index)
	uint64_t src = 0;               // (owned: the guest address in that image whose bytes `begin` holds; == begin unless
	                                // the bytes were moved here from another texture: OnTextureMove)
	uint64_t ready_tick = 0;        // (in the ring: a GPU copy writes them, done at that tick; 0: there now)
	uint32_t alloc = 0;             // (internal: the ring allocation)
};

// Host::thread_class: who is asking (nativeTex.cpp, top: who writes bytes back).
enum ThreadClass : int {
	ThreadNormal      = 0, // the render (GPU) thread, a game thread, any other thread that may wait for the render thread
	ThreadSpeculation = 1, // a speculative translation's thread
	ThreadRestricted  = 2, // may neither wait for the render thread nor record GPU work (async upload worker, a deferred
	                       // operation, a holder of the texture cache's lock off the render thread)
};

// What the texture cache provides (render thread functions; set once in its constructor).
struct Host {
	// The calling thread's ThreadClass (any thread).
	int (*thread_class)() = nullptr;
	// A speculation's thread: its translation refused (redone in order).
	void (*refuse_speculation)() = nullptr;
	// Runs fn(arg) where it may record GPU work and wait for the GPU, and returns when done: inline on the render thread;
	// from any other thread through the render thread (SendCommandSync; a speculation's thread with its recorder and
	// speculation state set aside and its speculation refused, as GpuResourceManager::HandleFault's read path does).
	// False (fn not run): a ThreadRestricted thread.
	bool (*run_render)(void (*fn)(void*), void* arg) = nullptr;
	// The page manager's read watchers (no access) of [vaddr, vaddr + size).
	void (*watch)(uint64_t vaddr, uint64_t size, bool on) = nullptr;
	// Render thread: the bytes of the owned pieces written into guest memory (from their images); returns those done.
	void (*materialize_owned)(const std::vector<Piece>& pieces) = nullptr;
	// Render thread: the guest range of image `image`/`serial` (false: gone, or its download is not supported).
	bool (*owned_extent)(uint64_t image, uint64_t serial, uint64_t* begin, uint64_t* size) = nullptr;
	// Render thread: all of that image's guest bytes copied by the GPU into the ring at `ring_offset` (its first byte
	// there), done at `*ready_tick`; false when it cannot.
	bool (*copy_owned)(uint64_t image, uint64_t serial, uint64_t ring_offset, uint64_t* ready_tick) = nullptr;
	// Render thread: the scheduler's tick now / whether a tick is done / wait for one.
	uint64_t (*current_tick)() = nullptr;
	bool (*tick_done)(uint64_t tick) = nullptr;
	void (*wait_tick)(uint64_t tick) = nullptr;
};

// Render thread, once (the texture cache's constructor): the ring allocated, the host functions set.
void Init(GraphicContext& graphics, CommandScheduler& scheduler, const Host& host);
// Enabled and the ring is there.
[[nodiscard]] bool Active() noexcept;
[[nodiscard]] Buffer* Ring() noexcept;

// Whether any piece intersects [addr, addr + size) (cheap when none is near).
[[nodiscard]] bool Overlaps(uint64_t addr, uint64_t size);
// The pieces intersecting [addr, addr + size), clipped to it, by address.
[[nodiscard]] std::vector<Piece> Collect(uint64_t addr, uint64_t size);
// The ring/undefined pieces `uploaded` (as Collect returned them, clipped to the image) now hold their bytes in image
// `image`/`serial` (uploaded by work at `tick`). A range replaced since Collect (a newer load) is left as it is.
void MarkOwned(const std::vector<Piece>& uploaded, uint64_t image, uint64_t serial, uint64_t tick);
// Render thread: every piece intersecting [addr, addr + size) (whole pieces; `serial` != 0: only those owned by that
// image) written into guest memory and dropped; their pages tracked normally again. `why` counts the fallback.
void Materialize(uint64_t addr, uint64_t size, const char* why, uint64_t serial = 0);
// A host write over [addr, addr + size) (texture cache InvalidateMemory): authoritative bytes there come back first.
void BeforeHostWrite(uint64_t addr, uint64_t size);
// Render thread: the pieces image `serial` (ImageId `image`, guest range [begin, begin + size)) holds, wherever they
// are, copied by the GPU back into the ring before the image goes (no wait); written to guest memory when the ring has
// no room.
void ImageGone(uint64_t image, uint64_t serial);
// Render thread: the pieces over [addr, addr + size) held by images other than `keep_serial`, back into the ring.
void Rehome(uint64_t addr, uint64_t size, uint64_t keep_serial);
// Before the emulator reads guest memory [addr, addr + size) (the backing-read functions of kernel/memory.cpp, the
// buffer cache's synchronization): authoritative bytes there written into guest memory first. Any thread.
void BeforeGuestRead(uint64_t addr, uint64_t size);
// GPU work wrote [addr, addr + size) (a buffer over it holds the bytes now): the pieces there dropped.
void Forget(uint64_t addr, uint64_t size);
// The exception filter's probe (any thread): a touch of an authoritative page. True: handled (the access goes again).
[[nodiscard]] bool OnFault(uint64_t address, uint64_t rip, const void* native_context, bool write);
// Render thread, once a flip: ring reclaim, spill policy, the 30 s log.
void Pump();
// Counters (the texture cache's native uploads).
void NoteNativeUpload(uint64_t bytes);
void NoteFallback(const char* why, uint64_t bytes);

} // namespace NativeTex
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVETEX_H_
