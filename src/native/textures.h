#pragma once
// (f5) Texture formats / detiling done once at load: the interface (design; no implementation in this build).
//
// Today a texture is discovered per draw from its T# (set validation td2 + tq2 0.47 ms p50), detiled on the GPU at each
// upload, and kept twice: the game's copy in RAM (VirtualMemo, ~2.26 GB committed) and the VkImage in VRAM.
// Natively the image is created when the game creates the texture (its factory, 0x8fe750 per grafico §3, to be
// re-proven before use), filled once from the game's data (detiled and format-converted once), and found by the
// game's texture object, not by the T# bits. With the streaming system's mips (ram15/ram19: the CPU does not read the
// streamed texels back) the RAM copy can then be released: -2.2 to -2.6 GB (F4 estimate).
//
// VRAM: the 8 GB budget (simular_vram_8gb=1) is a hard limit of the pool (grafico §7: 5.386 / 6.171 MiB p50 used).
#include <cstdint>

namespace NativeRender {

struct TextureAsset {
	uint64_t game_object  = 0; // the game's texture object (factory output)
	uint64_t guest_data   = 0; // texels as loaded (tiled)
	uint64_t guest_size   = 0;
	uint32_t width = 0, height = 0, depth = 1, mips = 1, layers = 1;
	uint32_t guest_format = 0; // PS5 data / number format
	uint32_t tile_mode    = 0;
};

struct TextureHandle {
	uint64_t image = 0; // VkImage
	uint64_t view  = 0; // VkImageView
	uint32_t resident_mips = 0;
};

class TextureLoader {
public:
	virtual ~TextureLoader() = default;
	// The game created a texture (factory hook): create the image, detile and convert once, upload.
	virtual TextureHandle Create(const TextureAsset& asset) = 0;
	// The streaming system brought mips [first, first + count) in: upload those only.
	virtual void          Stream(uint64_t game_object, uint32_t first_mip, uint32_t count) = 0;
	virtual void          Destroy(uint64_t game_object) = 0;
	[[nodiscard]] virtual TextureHandle Find(uint64_t game_object) const = 0;
	// The host format for a guest format, chosen once (no per-draw format decisions).
	[[nodiscard]] virtual uint32_t HostFormat(uint32_t guest_format) const = 0;
};

} // namespace NativeRender
