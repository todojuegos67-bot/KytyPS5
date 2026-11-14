#pragma once
// KYTY_ENGINE_TEX_REGISTRY=1 (default off): the textures as the GAME's engine declares them, so the texture cache can
// size an image from the engine's own texture instead of from what a bound T# seems to say.
//
// Filled by byte-checked guest hooks (loader/demonsSoulsEngineTex.cpp, 01.007.000):
//   - CTexture::FTextureFactory 0x8fe750 (every loaded texture) at its single epilogue: the engine's T# (+0xbc, 8 dwords,
//     the file's T# with the base the engine allocated), the width/height/depth/mips the engine decoded from it
//     (+0x264/+0x268/+0x26c/+0x274), and the bytes it reserved: a streamed texture's whole mip-chain run (last chunk's
//     offset + size), else the bytes it loaded ([+0x278], = the Alloc size of 0x8fed02);
//   - CTexture::Create 0x8ffed0 (render targets, TextureArrayManager pools, engine-made textures) at its single
//     epilogue: the declared w/h/depth/slices/mips (+0x264..+0x274) and the T# it built;
//   - removed when the engine releases the texture's memory: the three CTexture release calls of the deferred free
//     queue 0x903460 with [tex+0x98] (0x900c03, 0x900cc9, 0x900fa4).
// Read by the texture cache (renderer/pipeline/descriptors.cpp) on every uncached texture resolution.
#include <cstdint>

namespace Libs::Graphics::EngineTexRegistry {

enum class Source : uint8_t { Factory = 0, FactoryStreamed = 1, Create = 2, Count = 3 };

struct Entry {
	uint64_t base  = 0; // the engine T#'s base address (= CTexture+0x98, its allocation)
	uint64_t size  = 0; // bytes the engine reserved for it; 0 = not known in bytes (CTexture::Create)
	uint64_t owner = 0; // the CTexture*
	uint32_t tsharp[8] {};
	uint32_t width = 0, height = 0, depth = 0, layers = 0, mips = 0;
	uint8_t  engine_format = 0; // CTexture+0x60 (the engine's EFormat)
	uint8_t  flags         = 0; // CTexture+0x61 (&3 == 1: cube)
	int16_t  pool_id       = -1; // TextureArrayManager pool (0..10) whose CTexture this is; -1 none / not looked up
	Source   source        = Source::Factory;
};

// True once the loader's hooks are in (KYTY_ENGINE_TEX_REGISTRY=1 and every byte check passed).
[[nodiscard]] bool Enabled() noexcept;
void               SetEnabled(bool on) noexcept;

// Game threads (inside the engine's texture code): cheap, one short lock.
void Insert(const Entry& entry) noexcept;
void Release(uint64_t owner, uint64_t base) noexcept;

// The entry whose [base, base + max(size, 1)) holds `address`. False: none.
[[nodiscard]] bool Find(uint64_t address, Entry* out) noexcept;

// The TextureArrayManager pool whose CTexture `owner` is (global 0x2f684b0 -> [mgr] + id * 0x60 -> holder +0x20),
// looked up once per entry; -1 when none. `eboot_base` is the program's load address.
void  SetEbootBase(uint64_t base) noexcept;
int16_t PoolOf(uint64_t owner) noexcept;

// Texture-cache side. Called with the bound T# (8 dwords) of a sampled texture. On a hit (an engine texture at the
// T#'s base) and when the T#'s dwords 0..3 are the engine T#'s (ignoring MIN_LOD and BASE/LAST_LEVEL, which the engine
// changes while streaming), dwords 4..7 are replaced by the engine T#'s own: the extent the engine declared for that
// texture. Not when the T# is a layer view of it the engine made (BASE_ARRAY..DEPTH inside the engine's layers, words
// 5..7 the engine's): that view is the game's. Returns whether there was a hit (entry filled).
bool Resolve(uint32_t fields[8], Entry* entry) noexcept;

// Counters for the 30 s line.
enum class Count : uint32_t { Hit, Miss, Adopted, Clamp, Conflict, Count };
void Note(Count c) noexcept;
void NoteImage(uint64_t bytes, uint32_t width, uint32_t height, uint32_t layers, uint32_t levels) noexcept;
// Every 30 s (any thread): "EngineTex: ..." (only when enabled).
void MaybeLog() noexcept;
// A few detailed lines (descriptor vs engine T#) for conflicts and adoptions.
void Report(const char* what, uint64_t address, const uint32_t descriptor[8], const Entry& entry, uint32_t view_layers,
            uint32_t base_layer) noexcept;

} // namespace Libs::Graphics::EngineTexRegistry
