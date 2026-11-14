#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVETEXTUREPC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVETEXTUREPC_H_

// KYTY_NATIVE_TEXTURE_PC=1 (default off; off = no behaviour change): Demon's Souls' engine textures handled the way a PC
// port does it. The engine's own texture code is hooked (loader/demonsSoulsNativeTexturePc.cpp, byte-checked) and every
// event arrives here keyed by the engine's CTexture and the address its T# carries:
//
//   created      CTexture::Create 0x8ffed0 / FTextureFactory 0x8fe750 epilogues   OnTexture
//   loaded       FTextureFactory InputStream::Read 0x8fed17, pool layer 0x8fef15   OnLoad      (file bytes)
//   stream-in    SetCurrentStreamingChunk memcpy 0x901459                          OnLoad      (mip bytes)
//   stream-out   VirtualMemoryGPU ResizePages op 2 / shrink / run free             OnDrop
//   slice copy   TextureArrayManager texture -> pool layer memcpy 0x900f26         OnSliceCopy
//   destroyed    CTexture deferred free 0x903460 (0x900c03 / 0x900cc9 / 0x900fa4)  OnRelease
//
// Where the bytes live:
//   1. a load copies them ONCE into a host block (resource heap block -> host; the guest store is skipped);
//   2. the render thread (Flush: every image upload and every flip) moves each host block into its own device-local
//      buffer (one upload) and frees the host block: video memory holds the only copy from then on;
//   3. images are filled from those buffers by the GPU (BufferCache::ObtainBufferForImage hands the device buffer to the
//      detiler, or gathers several by GPU copies), so an image the texture cache drops and re-creates comes back from
//      VRAM, never from RAM; a streamed mip is its own device buffer at its own range;
//   4. a slice copy (0x900f26) makes the destination range share the source's device buffer (or host block): no byte is
//      moved at all, and the source's release keeps what the slice shares alive (reference counted);
//   5. the guest pages wholly inside a staged range are kept no-access (the page manager's read watchers). A CPU touch
//      faults (OnFault, the exception filter's probe) and any other emulator read calls BeforeGuestRead: the bytes are
//      read back from the GPU into guest memory once, the pages unwatched, counted as a CPU fallback. The partial edge
//      pages of a range (shared with other data) get their bytes written into guest memory at once.
//
// Every 30 s: "NativeTexturePC: ..." (textures, MiB loaded/streamed natively, host MiB held, VRAM MiB, uploads,
// upload ms/frame, CPU fallbacks).

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace Libs::Graphics::NativeTexturePc {

// KYTY_NATIVE_TEXTURE_PC (read once): 0 off; 1 = KEEP-GUEST (safe): the bytes go to VRAM for the images AND are
// written into guest memory once (no host copy kept, no page watching, nothing ever read back); a CPU/host write over a
// staged range (TextureCache::InvalidateMemory -> OnGuestWrite) drops the staged bytes there (the guest's are newer);
// slice copies run in guest memory (declined). 2 = VRAM-ONLY (the native goal): guest pages never written and kept
// no-access, read back from the GPU on a touch by the texture cache's own copy queue (textureCache.cpp
// NativeTexturePcQueue: own command pool + fence, graphics queue under queue_mutex, never the render thread), so any
// thread can read back; a failed read-back (device error) is an impossible case that stops with a FATAL line.
[[nodiscard]] int  Level() noexcept;
[[nodiscard]] bool Requested() noexcept; // Level() != 0
// The hooks are in and the path is live.
[[nodiscard]] bool Enabled() noexcept;
void               SetEnabled(bool on) noexcept;

// What the core needs from the emulator. Null members: not called (tests run with a fake device).
struct Host {
	// Game threads, after bytes for [addr, addr + size) are staged: the images over the range refreshed at next use.
	void (*staged)(uint64_t addr, uint64_t size) = nullptr;
	// Write `size` bytes at guest `addr` (the guest backing; any thread).
	void (*write_guest)(uint64_t addr, const void* data, uint64_t size) = nullptr;
	// Page manager read watchers of [addr, addr + size) on/off (page-aligned; any thread).
	void (*watch)(uint64_t addr, uint64_t size, bool on) = nullptr;
	// Render thread: a device-local buffer of `size` bytes, its fill from `data` recorded (done by upload_commit). Returns
	// its handle (0: none: the block stays on the host).
	uint64_t (*upload)(const void* data, uint64_t size) = nullptr;
	// Render thread: the uploads recorded since the last commit submitted together and waited for. False: none done (the
	// caller releases their buffers).
	bool (*upload_commit)() = nullptr;
	// Render thread, deferred until the GPU is done with it: the buffer gone.
	void (*release)(uint64_t handle) = nullptr;
	// Any thread: `size` bytes at `offset` of device buffer `handle` read back into `out` (waits for the GPU). False:
	// not possible on this thread.
	bool (*read_device)(uint64_t handle, uint64_t offset, void* out, uint64_t size) = nullptr;
	// Any thread: the device-local video memory in use and the budget (the OS's), bytes. False: unknown.
	bool (*vram)(uint64_t* usage, uint64_t* budget) = nullptr;
	uint64_t page_size = 4096;
};
// Sets the non-null members (the loader sets the guest side, the texture cache the device side).
void SetHost(const Host& host) noexcept;
// The device side is set: the pages of everything staged before are watched now.
void HostReady() noexcept;

enum class Kind : uint8_t { Loaded = 0, Streamed = 1, Created = 2 };

struct Texture {
	uint64_t owner = 0; // CTexture*
	uint64_t base  = 0; // its T# base (= [CTexture+0x98])
	uint64_t size  = 0; // bytes the engine reserved for it (FTextureFactory); 0 = unknown (CTexture::Create)
	uint32_t width = 0, height = 0, depth = 0, layers = 0, mips = 0;
	uint8_t  format = 0;
	Kind     kind   = Kind::Loaded;
	bool     slice_copy_candidate = false; // [tex+0x65] == 1 after AddTexture: loaded alone, copied into a pool later
};

// ---- Game threads (the hooks) --------------------------------------------------------------------------------------

void OnTexture(const Texture& texture) noexcept;
// True: staged here, the caller must NOT write guest memory.
[[nodiscard]] bool OnLoad(uint64_t owner, uint64_t owner_base, uint64_t dst, const void* src, uint64_t size,
                          bool mip) noexcept;
void OnDrop(uint64_t addr, uint64_t size) noexcept;
// True: done here (dst shares src's bytes; guest bytes of src read only outside engine texture memory). False: not
// ours, the caller copies guest memory.
[[nodiscard]] bool OnSliceCopy(uint64_t dst, uint64_t src, uint64_t size,
                               const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept;
void OnRelease(uint64_t owner, uint64_t base) noexcept;

// ---- Emulator side ---------------------------------------------------------------------------------------------------

[[nodiscard]] bool Overlaps(uint64_t addr, uint64_t size) noexcept;

// Render thread: every host block moved into its device buffer (host blocks freed), device buffers no one uses any more
// released.
void Flush() noexcept;

// How [addr, addr + size) is made, for an image upload (after Flush).
struct Part {
	uint64_t begin = 0, end = 0;
	enum : uint8_t { Device, Host, Zero, Guest } kind = Guest;
	uint64_t       handle = 0, offset = 0; // Device: the buffer and the offset of `begin` in it
	const uint8_t* host   = nullptr;       // Host: the bytes (alive while the plan is: `keep`)
	std::shared_ptr<const void> keep;      // holds the buffer / block for the plan's life
};
[[nodiscard]] std::vector<Part> Plan(uint64_t addr, uint64_t size) noexcept;
void                            NoteUpload(uint64_t bytes, uint64_t ns, bool gathered) noexcept;

// Tests and the CPU path: the bytes of [addr, addr + size) into `out` (device parts read back with read_device).
[[nodiscard]] bool Fill(uint64_t addr, uint64_t size, void* out,
                        const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept;

// A CPU or host write over [addr, addr + size) (TextureCache::InvalidateMemory): level 1: staged bytes there dropped
// (guest memory is authoritative there now). Level 2: the bytes there into guest memory first (then dropped).
void OnGuestWrite(uint64_t addr, uint64_t size) noexcept;
// Tests: the level as if from the environment.
void SetLevelForTests(int level) noexcept;
// Before the emulator reads guest memory [addr, addr + size) another way: the bytes there into guest memory once.
void BeforeGuestRead(uint64_t addr, uint64_t size) noexcept;
// The exception filter's probe: a CPU touch of a no-access page of ours. True: handled (the access goes again).
[[nodiscard]] bool OnFault(uint64_t address) noexcept;
// Once a flip (render thread): Flush, frame count, the 30 s log line.
void Pump() noexcept;

// KYTY_VRAM_BUDGET (read once): "auto"/"1" on, "0"/"off" off; unset: the device's default (SetVramBudgetDefault: on
// for a GPU short of video memory, < 12 GiB). On: near the device-local budget, least recently used engine textures'
// device buffers are demoted to guest memory (level 1 style), so the images are their only video memory copy (Pump).
[[nodiscard]] int  VramBudgetMode() noexcept; // -1 unset, 0 off, 1 auto
void               SetVramBudgetDefault(bool on) noexcept;
[[nodiscard]] bool VramBudgetOn() noexcept;

// KYTY_NATIVE_TEXTURE_PC_SINGLE=1 (read once; needs level 2): device buffers released once their image is built (Pump).
[[nodiscard]] bool SingleCopy() noexcept;

struct Stats {
	uint64_t textures = 0, textures_loaded = 0, textures_created = 0, released = 0;
	uint64_t loads = 0, load_bytes = 0, mips = 0, mip_bytes = 0, slice_copies = 0, slice_bytes = 0, slice_guest_reads = 0;
	uint64_t dropped_bytes = 0, host_bytes = 0, host_blocks = 0, vram_bytes = 0, vram_buffers = 0;
	uint64_t to_vram = 0, to_vram_bytes = 0, upload_fail = 0;
	uint64_t fills = 0, fill_bytes = 0, fill_ns = 0, gathered = 0, frames = 0;
	uint64_t fallbacks = 0, fallback_bytes = 0, faults = 0, fallback_refused = 0, declined = 0, watched_bytes = 0;
	uint64_t guest_writes = 0, guest_write_bytes = 0;
};
[[nodiscard]] Stats GetStats() noexcept;
void                ResetForTests() noexcept;

} // namespace Libs::Graphics::NativeTexturePc

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVETEXTUREPC_H_
