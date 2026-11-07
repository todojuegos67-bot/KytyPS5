// Demon's Souls' texture streamer (BPE TextureManager; 01.007.000 eboot+0xac5b70 updates a texture's record each frame)
// streams a texture's levels in parts, the smallest first: a 2048x2048 texture as levels 3.. (the tail), then 2, 1 and 0,
// a part each (the texture's part table at + 0x70, each part's first level at its + 0). From the level the frame wants
// (one more once that one is in) the update picks the first level to hold, r15, then lets textureForceWantTopMip
// override it. With KYTY_TEXTURE_REDUCE the GPU holds a large block-compressed texture without its 1 or 2 largest levels
// (descriptors.cpp): their parts were streamed into the game's memory and never read. Here the first level to hold stays
// at or past the levels the GPU leaves out, for the textures it leaves them out of: one layer (+ 0x26c), more levels than
// that (+ 0x274), at least 1024 texels on a side (+ 0x264, + 0x268), and a T# (+ 0xbc) of a 2D image, of the texture's
// height, in a block-compressed format. The GPU holds the same levels as before (from the T#'s first level). Textures
// the game keeps whole (TextureManager::CacheHighMips, + 0x6c > 0) do not come here.
#include "loader/demonsSoulsTextureMips.h"

#include "common/virtualMemory.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "local/local-platform.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>

namespace Loader::DemonsSoulsTextureMips {
namespace {
constexpr uint64_t PageSize = 0x4000, CaveOffset = 0x8000000 + 8 * PageSize;

// 01.007.000 eboot+0xac5c8a (rdx: the texture): test r15d, r15d; setg r8b; test ecx, ecx; mov [r14 + 0x18], ecx; setne
// r9b; and r9b, sil; and r9b, r8b; movzx esi, r9b; mov r9, [rdx + 0x78]; sub r15d, esi (the level more); mov rsi,
// [rip + d] (the override's value pointer); mov esi, [rsi]; test esi, esi; cmovns r15d, esi.
constexpr const char* Pick = "45 85 ff 41 0f 9f c0 85 c9 41 89 4e 18 41 0f 95 c1 41 20 f1 45 20 c1 41 0f b6 f1 4c 8b 4a 78 "
                             "41 29 f7 48 8b 35 ?? ?? ?? ?? 8b 36 85 f6 44 0f 49 fe";
constexpr uint64_t SiteOffset = 31; // sub r15d, esi (3 bytes) and mov rsi, [rip + d] (7) move to the stub
constexpr uint32_t SiteSize   = 10;
constexpr uint32_t Formats    = 512; // the T#'s format field: 9 bits

bool SavesRam() {
	const auto ram = LocalPlatform::EmulatorMemory();
	return ram != 0 && ram <= (uint64_t {16} << 30u);
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size > CaveOffset || program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	// Only with 16 GiB of RAM or less for the emulator (LocalPlatform::EmulatorMemory), where the ~1 GiB those levels take
	// counts. Left out, they make the buffers BDA shaders read grow with the worlds the game loads, from ~2 GiB to 3-3.4 in
	// 30-40 loads against ~2 streamed whole (why is not known): past an 8 GB GPU's video memory budget for good (11-07, six
	// worlds by archstone travel, 48 loads an hour: Boletaria 48 fps against 53 streamed whole on a PC with more RAM,
	// standing 49 against 60). KYTY_TEXTURE_REDUCE_GAME=0: streamed whole on any PC; =1: left out on any PC.
	const char*    game   = std::getenv("KYTY_TEXTURE_REDUCE_GAME");
	const uint32_t levels = Libs::Graphics::GameTextureReduceLevels();
	if (levels == 0) return;
	if (game != nullptr && *game != '\0' ? game[0] == '0' : !SavesRam()) {
		if (game == nullptr || *game == '\0')
			std::printf("Demon's Souls texture streaming: textures streamed whole (more than 16 GiB of RAM; "
			            "KYTY_TEXTURE_REDUCE_GAME=1 leaves out the levels the GPU leaves out)\n");
		return;
	}
	const auto pick = GuestCode::FindUnique(*program, GuestCode::Pattern(Pick));
	if (!pick) {
		std::printf("Demon's Souls texture streaming: the streamer is not of a known build; textures streamed whole\n");
		return;
	}
	const auto site = *pick + SiteOffset;
	int32_t    displacement = 0;
	std::memcpy(&displacement, reinterpret_cast<const void*>(site + 6), sizeof(displacement));
	const uint64_t override_pointer = site + SiteSize + static_cast<int64_t>(displacement);
	const auto     requested        = program->base_vaddr + CaveOffset;
	const auto     allocated        = Libs::LibKernel::Memory::AllocateRuntimeMemory(
        requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_texture_mips", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// The block-compressed formats (the GPU's ImageInfo::IsBlock), a bit each, at the page's end.
	auto* formats = reinterpret_cast<uint8_t*>(allocated + PageSize - Formats / 8);
	std::memset(formats, 0, Formats / 8);
	for (uint32_t format = 0; format < Formats; format++) {
		if (Libs::Graphics::Prospero::BlockCompressedBytesPerBlock(static_cast<Libs::Graphics::Prospero::BufferFormat>(format)) != 0)
			formats[format / 8] |= static_cast<uint8_t>(1u << (format % 8));
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize - Formats / 8, reinterpret_cast<void*>(allocated));
	Xbyak::Label        keep;
	// The level more, then the first level held past the levels left out. (esi and r8 are free: the code after the site
	// loads both before a use.)
	c.sub(r15d, esi);
	c.cmp(r15d, levels);
	c.jge(keep);
	c.cmp(dword[rdx + 0x26c], 1);
	c.jne(keep);
	c.cmp(dword[rdx + 0x274], levels);
	c.jbe(keep);
	c.mov(esi, dword[rdx + 0x264]);
	c.mov(r8d, dword[rdx + 0x268]);
	c.cmp(esi, r8d);
	c.cmovb(esi, r8d);
	c.cmp(esi, 1024);
	c.jb(keep);
	c.mov(esi, dword[rdx + 0xc4]); // T# dword 2: the height less 1 at bits 14..27
	c.shr(esi, 14);
	c.and_(esi, 0x3fff);
	c.inc(esi);
	c.cmp(esi, r8d);
	c.jne(keep);
	c.mov(esi, dword[rdx + 0xc8]); // dword 3: the type at bits 28..31 (9: 2D)
	c.shr(esi, 28);
	c.cmp(esi, 9);
	c.jne(keep);
	c.mov(esi, dword[rdx + 0xc0]); // dword 1: the format at bits 20..28
	c.shr(esi, 20);
	c.and_(esi, Formats - 1);
	c.mov(r8, reinterpret_cast<uint64_t>(formats));
	c.bt(dword[r8], esi);
	c.jnc(keep);
	c.mov(r15d, levels);
	// The moved load of the override's value pointer (its address now absolute) and back.
	c.L(keep);
	c.mov(rsi, override_pointer);
	c.mov(rsi, qword[rsi]);
	c.jmp(reinterpret_cast<const void*>(site + SiteSize));
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(allocated, c.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	// (The bytes after the jump are never reached: nothing jumps into the moved instructions.)
	uint8_t code[SiteSize];
	std::memset(code, 0xcc, sizeof(code));
	code[0]          = 0xe9;
	const auto jump  = static_cast<int32_t>(allocated - (site + 5));
	std::memcpy(code + 1, &jump, sizeof(jump));
	std::memcpy(reinterpret_cast<void*>(site), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(site, sizeof(code));
	std::printf("Demon's Souls texture streaming: large block-compressed textures streamed without their %u largest "
	            "level(s), as the GPU holds them (eboot+0x%llx; KYTY_TEXTURE_REDUCE_GAME=0 streams them whole)\n",
	            levels, static_cast<unsigned long long>(site - program->base_vaddr));
#endif
}

} // namespace Loader::DemonsSoulsTextureMips
