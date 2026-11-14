// KYTY_NATIVE_TEXTURE_PC: see demonsSoulsNativeTexturePc.h and graphics/host_gpu/nativeTexturePc.h.
// Static facts (01.007.000 ELF vaddrs; lab/static/facts/ds-native-texture-pc.txt, checked against the eboot):
//   stream-in (0x901330, SetCurrentStreamingChunk): 0x9013cb r12 = [r14+0x98] + [r13+8] (r14 = the CTexture, r13 = the
//     chunk), 0x9013d6 r15d = [r13+4]; 0x9013fa Realloc(r12, r15) commits; r14 is not written up to 0x901459;
//     0x901450 mov rdi,r12; mov rsi,rax; mov rdx,r15; 0x901459 call memcpy (0x1e73d00).
//   AddTexture 0xac4980: 0xac4b61 [tex+0x68] = layer; 0xac4b9c [tex+0x65] = 1 when the pool's flag bit 0 is clear (the
//     texture is loaded into its own memory: FTextureFactory 0x8fec98 je 0x8fece4 -> Alloc 0x8fed02), = 2 when loaded
//     straight into the pool slice (0xac4ba3); those of [+0x65] == 1 are the ones 0xac5190 -> 0x900e50 later copies.
#include "loader/demonsSoulsNativeTexturePc.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/nativeTexturePc.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsEngineTex.h"
#include "loader/demonsSoulsNativeTex.h"
#include "loader/demonsSoulsNativeTexStream.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace Loader::DemonsSoulsNativeTexturePc {

namespace {

namespace Pc = Libs::Graphics::NativeTexturePc;

constexpr uint64_t MipCopyCall = 0x901459, MemcpyStub = 0x1e73d00;
constexpr uint64_t CaveOffset = 0x80d8000, CaveSize = 0x1000;
constexpr uint64_t TexAlloc = 0x98;

uint64_t g_base = 0;

uint64_t Load64(uint64_t a) {
	uint64_t v = 0;
	std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof(v));
	return v;
}

bool Bytes(uint64_t address, std::initializer_list<uint8_t> bytes) {
	if (!Libs::Graphics::HostMemoryRangeIsReadable(address, bytes.size())) return false;
	return std::memcmp(reinterpret_cast<const void*>(address), bytes.begin(), bytes.size()) == 0;
}

bool CallsTo(uint64_t site, uint64_t target) {
	if (!Libs::Graphics::HostMemoryRangeIsReadable(site, 5)) return false;
	const auto* p = reinterpret_cast<const uint8_t*>(site);
	if (p[0] != 0xe8) return false;
	int32_t rel = 0;
	std::memcpy(&rel, p + 1, sizeof(rel));
	return site + 5 + static_cast<int64_t>(rel) == target;
}

// The stream-in mip copy (memcpy's contract: returns dst). rcx = the CTexture (r14).
void* KYTY_SYSV_ABI MipCopy(void* destination, const void* source, size_t size, uint64_t texture) {
	const auto to = reinterpret_cast<uint64_t>(destination);
	if (size != 0 && Pc::OnLoad(texture, texture != 0 ? Load64(texture + TexAlloc) : 0, to, source, size, true))
		return destination;
	const bool prepared = size != 0 && Libs::LibKernel::Memory::PrepareHostOverwrite(to, size);
	std::memcpy(destination, source, size);
	if (size != 0) Libs::LibKernel::Memory::FinishHostOverwrite(to, size, prepared);
	return destination;
}

const char* InstallMipHook(uint64_t base) {
	if (!Bytes(base + 0x9013cb, {0x4d, 0x8b, 0xa6, 0x98, 0x00, 0x00, 0x00, 0x4d, 0x03, 0x65, 0x08, 0x45, 0x8b, 0x7d, 0x04}) ||
	    !Bytes(base + 0x901450, {0x4c, 0x89, 0xe7, 0x48, 0x89, 0xc6, 0x4c, 0x89, 0xfa}))
		return "stream-in bytes differ";
	if (!CallsTo(base + MipCopyCall, base + MemcpyStub)) return "0x901459 is not the memcpy call (KYTY_NATIVE_MEM=3 owns it?)";
	const uint64_t cave = Libs::LibKernel::Memory::AllocateRuntimeMemory(base + CaveOffset, CaveSize,
	                                                                     Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                     "demons_souls_native_texture_pc", true);
	if (cave != base + CaveOffset) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "no cave memory at its address";
	}
	const int64_t d = static_cast<int64_t>(cave) - static_cast<int64_t>(base + MipCopyCall + 5);
	if (d != static_cast<int32_t>(d)) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "cave out of rel32 reach";
	}
	// mov rcx,r14; mov rax,imm64; jmp rax (rcx: no argument of memcpy, caller-saved; the return address is the original)
	std::array<uint8_t, 15> stub {0x4c, 0x89, 0xf1, 0x48, 0xb8};
	const auto              fn = reinterpret_cast<uint64_t>(&MipCopy);
	std::memcpy(stub.data() + 5, &fn, 8);
	stub[13] = 0xff;
	stub[14] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(cave), stub.data(), stub.size());
	if (!Common::VirtualMemory::FlushInstructionCache(cave, stub.size())) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "instruction cache flush failed";
	}
	std::array<uint8_t, 5> call {0xe8};
	const auto             rel = static_cast<int32_t>(d);
	std::memcpy(call.data() + 1, &rel, 4);
	std::memcpy(reinterpret_cast<void*>(base + MipCopyCall), call.data(), call.size());
	if (!Common::VirtualMemory::FlushInstructionCache(base + MipCopyCall, call.size())) return "instruction cache flush failed (site)";
	return nullptr;
}

// ---- consumers of the shared hooks ----

bool LoadConsumer(const ::Loader::NativeTex::TextureBytes& b) noexcept {
	const uint64_t owner_base = b.texture != 0 ? Load64(b.texture + TexAlloc) : 0;
	return Pc::OnLoad(b.texture, owner_base, b.guest_dst, b.host_src, b.size, false);
}
void StreamRelease(uint64_t addr, uint64_t size) noexcept { Pc::OnDrop(addr, size); }
bool StreamMove(uint64_t dst, uint64_t src, uint64_t size) noexcept {
	return Pc::OnSliceCopy(dst, src, size, [](uint64_t at, void* to, uint64_t n) {
		return Libs::LibKernel::Memory::TryReadBackingToHost(at, to, n);
	});
}

void Staged(uint64_t addr, uint64_t size) {
	// The images over the range refreshed (from Fill) at their next use.
	(void)Libs::LibKernel::Memory::TryPrepareHostImageWrite(addr, size);
}
void WriteGuest(uint64_t addr, const void* data, uint64_t size) {
	if (Pc::Level() == 1) {
		// The guest's copy of a load, as a file read's host write: the trackers told before and after.
		const bool prepared = Libs::LibKernel::Memory::PrepareHostOverwrite(addr, size);
		Libs::LibKernel::Memory::WriteBacking(addr, data, size, "native-texture-pc");
		Libs::LibKernel::Memory::FinishHostOverwrite(addr, size, prepared);
		return;
	}
	Libs::LibKernel::Memory::WriteBacking(addr, data, size, "native-texture-pc fallback");
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (!Pc::Requested()) return;
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	const char* why = nullptr;
	if (::Loader::NativeTex::Requested()) why = "KYTY_NATIVE_TEX / KYTY_ENGINE_GPU_NATIVE is on (the emulator-side ring path)";
	else if (!::Loader::NativeTex::Installed()) why = "the texture-load hooks (0x8fed17 / 0x8fef15) are not installed";
	else if (DemonsSoulsNativeTexStream::g_installed.load() != 15u) why = "the streaming hooks are not all installed";
	else if (!DemonsSoulsEngineTex::HooksInstalled()) why = "the CTexture create/factory/release hooks are not installed";
	g_base = program->base_vaddr;
	if (why == nullptr) why = InstallMipHook(g_base);
	if (why != nullptr) {
		std::printf("[native-texture-pc] KYTY_NATIVE_TEXTURE_PC=1 NOT applied: %s; textures load into guest memory as before\n", why);
		std::fflush(stdout);
		return;
	}
	Pc::Host host;
	host.staged      = &Staged;
	host.write_guest = &WriteGuest;
	Pc::SetHost(host);
	::Loader::NativeTex::SetConsumer(&LoadConsumer);
	::Loader::NativeTex::SetStreamConsumer(&StreamRelease, &StreamMove, nullptr, &StreamRelease);
	Pc::SetEnabled(true);
	if (Pc::Level() == 2)
		std::printf("[native-texture-pc] WARNING: KYTY_NATIVE_TEXTURE_PC=2 (VRAM only, first field test): engine texture "
		            "pages are not written and are no-access; a CPU touch reads them back from the GPU (counted as CPU "
		            "fallbacks). If textures look wrong or the game stops with 'NativeTexturePC: FATAL', use "
		            "KYTY_NATIVE_TEXTURE_PC=1 and send the log.\n");
	std::printf("[native-texture-pc] KYTY_NATIVE_TEXTURE_PC=1: engine textures registered at CTexture::Create/FTextureFactory, "
	            "file bytes (eboot+0x8fed17/0x8fef15) and streamed mips (eboot+0x%llx) staged on the host and uploaded to the "
	            "images from there; guest texture pages are not written; slice copies (0x900f26) done host-side; stream-out "
	            "and destroy drop the host bytes\n",
	            static_cast<unsigned long long>(MipCopyCall));
	std::fflush(stdout);
#else
	(void)program;
#endif
}

} // namespace Loader::DemonsSoulsNativeTexturePc
