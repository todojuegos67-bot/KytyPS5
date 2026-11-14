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
// The bytes the game would have stored into guest texture memory are copied ONCE (the resource heap block -> a host
// staging block owned here) and the guest store is skipped: the guest pages are reserved/committed by the engine's own
// allocator calls as before, but never written, so with lazy commit they never take RAM. The GPU never reads texture
// memory directly: the engine copies the 32-byte T# (base = [CTexture+0x98] >> 8) by value into its descriptor tables
// (e.g. 0x905fa1.., 0x91dafb.., 0x995..), so the renderer resolves a bound T# by its base address and fills the image
// from here (Fill: host staging -> Vulkan staging -> detile -> VkImage), never from guest memory.
//
// The host block of a texture is kept until the engine streams that range out or destroys the texture (a PC game's
// "managed" copy): the texture cache may drop and re-create an image at any time, and 0x900f26 copies a loaded texture
// into a pool layer on the CPU after load (slice copies are done here, from the host blocks, never from guest memory).
// Any other emulator read of the guest bytes (BeforeGuestRead) writes them into guest memory first: a counted CPU
// fallback.
//
// Every 30 s: "NativeTexturePC: ..." (textures, MiB loaded/streamed natively, guest MiB never written, uploads served,
// upload ms/frame, CPU fallbacks).

#include <cstdint>
#include <functional>

namespace Libs::Graphics::NativeTexturePc {

// KYTY_NATIVE_TEXTURE_PC=1 (read once).
[[nodiscard]] bool Requested() noexcept;
// The hooks are in and the path is live.
[[nodiscard]] bool Enabled() noexcept;
void               SetEnabled(bool on) noexcept;

// What the core needs from the emulator (set once, before SetEnabled(true)). Null members: not called.
struct Host {
	// Bytes for [addr, addr + size) are now staged here (a load, a mip, a slice copy): images over the range must be
	// refreshed (the texture cache's dirty marking). Called on the game's thread, outside the core's lock.
	void (*staged)(uint64_t addr, uint64_t size) = nullptr;
	// CPU fallback: write `size` bytes at guest `addr` (the guest backing; the trackers' notifications are the host's).
	void (*write_guest)(uint64_t addr, const void* data, uint64_t size) = nullptr;
};
void SetHost(const Host& host) noexcept;

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

// The engine finished making a texture (factory or Create epilogue).
void OnTexture(const Texture& texture) noexcept;
// The game would store `size` bytes from host-readable `src` at guest `dst` (inside the texture `owner` whose memory
// starts at `owner_base`; 0 when not known). True: staged here, the caller must NOT write guest memory.
[[nodiscard]] bool OnLoad(uint64_t owner, uint64_t owner_base, uint64_t dst, const void* src, uint64_t size,
                          bool mip) noexcept;
// The engine gave [addr, addr + size) of texture memory back (stream-out, run shrink, run free): staged bytes dropped.
void OnDrop(uint64_t addr, uint64_t size) noexcept;
// The engine copies `size` bytes from guest `src` to guest `dst` (0x900f26). True: done here (dst staged from src's
// host bytes; guest bytes of src read only where nothing is staged and src is not an engine texture's memory).
// False: not ours (nothing staged near src or dst): the caller copies guest memory as before.
[[nodiscard]] bool OnSliceCopy(uint64_t dst, uint64_t src, uint64_t size,
                               const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept;
// The engine destroys the texture (deferred free of [tex+0x98]).
void OnRelease(uint64_t owner, uint64_t base) noexcept;

// ---- Emulator side ---------------------------------------------------------------------------------------------------

// Cheap: whether [addr, addr + size) touches staged bytes or an engine texture's memory this path owns.
[[nodiscard]] bool Overlaps(uint64_t addr, uint64_t size) noexcept;
// The image bytes of [addr, addr + size) into host memory `out`: staged bytes where staged; zeros where it is the memory
// of an engine texture with nothing staged (undefined content: a streamed-out mip, a reserved run); elsewhere
// read_guest(at, to, n) (memory this path does not own). False when read_guest failed. Counts an upload served.
[[nodiscard]] bool Fill(uint64_t addr, uint64_t size, void* out,
                        const std::function<bool(uint64_t, void*, uint64_t)>& read_guest) noexcept;
// Before the emulator reads guest memory [addr, addr + size) some other way (a buffer over it, a hash): staged bytes
// there written into guest memory once (counted as a CPU fallback).
void BeforeGuestRead(uint64_t addr, uint64_t size) noexcept;
// Once a flip (render thread): frame count for ms/frame, and the 30 s log line.
void Pump() noexcept;

struct Stats {
	uint64_t textures = 0, textures_loaded = 0, textures_created = 0, released = 0;
	uint64_t loads = 0, load_bytes = 0, mips = 0, mip_bytes = 0, slice_copies = 0, slice_bytes = 0, slice_guest_reads = 0;
	uint64_t dropped_bytes = 0, held_bytes = 0, held_chunks = 0;
	uint64_t fills = 0, fill_bytes = 0, fill_ns = 0, frames = 0;
	uint64_t fallbacks = 0, fallback_bytes = 0, declined = 0;
};
[[nodiscard]] Stats GetStats() noexcept;
// Tests: everything forgotten (not thread-safe against hooks).
void ResetForTests() noexcept;

} // namespace Libs::Graphics::NativeTexturePc

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_NATIVETEXTUREPC_H_
