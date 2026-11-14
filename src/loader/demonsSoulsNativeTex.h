#pragma once
// KYTY_NATIVE_TEX=1 (off by default): the texture LOAD path of Demon's Souls (01.007.000) handed to the host, so a
// native-port upload (file bytes -> staging -> VRAM) can take the bytes without them having to land in guest RAM.
// Static notes: /home/claude/static/native-tex/A-load-path.md.
//
// What is hooked (each a 5-byte `call InputStream::Read 0x813130` pointed at a code cave; every byte around it is
// checked first, any mismatch: nothing is installed):
//   Site::TextureFile  0x8fed17 in CTexture::FTextureFactory (0x901240 -> 0x8fe750): the whole texture of a
//                      non-streamed texture (Alloc 0x8fed02, GPU context 0x2ea0d18, 64 KiB), or the resident part (the
//                      first mip-table entry) of a streamable one (Realloc 0x8fec1d / 0x8fec4f, VirtualMemoryGPU 0x2f64438).
//                      `texture` = the new CTexture (rbx).
//   Site::PoolLayer    0x8fef15 in 0x8fedc0: a texture that goes into a slice of a TextureArrayManager pool texture
//                      (same tile mode): `texture` = the POOL CTexture (r15); guest_dst = pool base + layer * slice size.
// Not hooked (stays as it was): the pool path with a different tile mode (0x8feeaf reads into a heap temp, then the
// CPU tiler 0x1d2fb70 at 0x8feed1 writes the pool slice), the fallback-texture memset 0x8ff001, streamed mips 0x901459
// (KYTY_NATIVE_MEM=3 owns that call).
//
// The stream FTextureFactory reads from is a memory InputStream (vtable 0x25ad5d0, ResourceFactory::CreateResource
// 0x8a9c66) over the resource's whole file data, already read/decompressed into a CPU heap block (0x8a9c03/0x8a9c2a):
// InputStream::Read there is one memcpy (0x8132b2) from [stream+0x120] + [stream+0x108]. So the hook can hand the host
// a host-readable pointer to exactly the bytes the game would have copied, and the consumer copies them once (heap
// block -> its staging buffer) instead of heap block -> guest texture memory -> upload.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace Loader {
struct Program;
namespace NativeTex {

// KYTY_NATIVE_TEX=1 or KYTY_ENGINE_GPU_NATIVE=1 (both default off): the engine's texture memory (VirtualMemoryGPU runs, the
// texture loads and streamed mips the hooks see) is staged on the GPU and its guest pages are left unwritten.
inline bool Requested() noexcept {
	for (const char* name: {"KYTY_ENGINE_GPU_NATIVE", "KYTY_NATIVE_TEX"}) {
		const char* v = std::getenv(name);
		if (v != nullptr && std::strcmp(v, "1") == 0) return true;
	}
	return false;
}
inline const char* RequestedName() noexcept {
	const char* v = std::getenv("KYTY_ENGINE_GPU_NATIVE");
	return v != nullptr && std::strcmp(v, "1") == 0 ? "KYTY_ENGINE_GPU_NATIVE=1" : "KYTY_NATIVE_TEX=1";
}

// The hooks themselves: also for KYTY_NATIVE_TEXTURE_PC=1 (graphics/host_gpu/nativeTexturePc.h, its own consumers).
inline bool HooksRequested() noexcept {
	if (Requested()) return true;
	const char* v = std::getenv("KYTY_NATIVE_TEXTURE_PC");
	return v != nullptr && (v[0] == '1' || v[0] == '2');
}

enum class Site : uint32_t {
	TextureFile = 0, // 0x8fed17
	PoolLayer   = 1, // 0x8fef15
	Count       = 2,
};

struct TextureBytes {
	uint64_t    guest_dst = 0;       // where the game writes the bytes (texture memory)
	uint64_t    size      = 0;       // bytes
	const void* host_src  = nullptr; // the same bytes, host-readable (the resource heap block); never null when offered
	uint64_t    stream    = 0;       // the guest InputStream* (memory stream), for diagnostics only
	uint64_t    texture   = 0;       // CTexture* (Site::TextureFile: the new texture; Site::PoolLayer: the pool texture)
	Site        site      = Site::TextureFile;
};

// The emulator side (an upload path that takes the bytes). Called on the loading thread (a game job; possibly a fiber
// with a small stack: keep it short, no deep recursion), before the game's copy. Return true to take the bytes: the
// guest destination is then NOT written (the stream is advanced as the copy would have advanced it), and from then on
// the consumer owns the obligation that any CPU read of [guest_dst, guest_dst + size) sees these bytes (see
// A-load-path.md: 0x900e50 memcpys a loaded texture into a pool slice on the CPU). Return false: the original copy runs.
using Consumer = bool (*)(const TextureBytes& bytes) noexcept;
void SetConsumer(Consumer consumer) noexcept;
// Whether the hooks are installed (KYTY_NATIVE_TEX=1 and every byte check passed).
[[nodiscard]] bool Installed() noexcept;

// The callback the hooks call. Offers the bytes to the consumer when one is set and the bytes are host-readable;
// otherwise (no consumer, or it declines) performs the original write: the game's own InputStream::Read(stream,
// guest_dst, size), the same call the hook replaced (identical behaviour). Returns true when the consumer took them.
bool OnTextureBytes(uint64_t guest_dst, const TextureBytes& src, uint64_t size) noexcept;

struct Stats {
	uint64_t calls[static_cast<uint32_t>(Site::Count)];
	uint64_t bytes[static_cast<uint32_t>(Site::Count)];
	uint64_t consumed_calls, consumed_bytes;
	uint64_t not_memory_stream; // offered nothing: the stream was not a whole-in-memory stream (original Read)
};
[[nodiscard]] Stats GetStats() noexcept;

// Load time (the runtime linker's patch list, after relocation, before any module starts).
void Install(Program* program);

} // namespace NativeTex
} // namespace Loader
