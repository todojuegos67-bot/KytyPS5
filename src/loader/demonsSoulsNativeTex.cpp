// KYTY_NATIVE_TEX: see demonsSoulsNativeTex.h. Static facts (01.007.000 eboot, ELF vaddrs; read instruction by
// instruction, /home/claude/static/native-tex/A-load-path.md):
//   FTextureFactory 0x901240: 0x9012ac call 0x8fe750 (rdi = [ctx+8] InputStream*, rsi = [ctx+0x10] end position,
//                   rcx = 0x2ea0d18 GPU context)
//   0x8fe750        0x8feafa Read(stream, CTexture+0xbc, 0x20) (the T#); r13 = end - [stream+0x130] (data size)
//                   streamable (mip entries >= 2): 0x8fec1d Realloc(0, total, VirtualMemoryGPU) -> CTexture+0x98,
//                     0x8fec4f Realloc(base + e0.offset, e0.size) commits it; 0x8fec54 rdi = stream, rsi = r12 =
//                     base + e0.offset, rdx = r13 = e0.size; jmp 0x8fed14
//                   pool (0xac4980 found a TextureArrayManager slot): 0x8fecad call 0x8fedc0, no Read here
//                   otherwise: 0x8fed02 Alloc(r13, GPU context, 64 KiB) -> CTexture+0x98 (r12 = rbx + 0x98)
//                   0x8fed17 call 0x813130 (rdi = stream, rsi = destination, rdx = size); rbx = the CTexture
//   0x8fedc0        r15 = the pool CTexture (0x8fedf4 mov r15,rdi; not written again on the 0x8fef02 path)
//                   0x8fef02..0x8fef15 rsi = pool base + layer * slice size, rdx = slice(s) size, call 0x813130
//   InputStream::Read 0x813130: avail = [s+0x110] - [s+0x108]; size <= avail: memcpy(dst, [s+0x120] + [s+0x108], size)
//                   (0x8132b2), [s+0x108] += size, [s+0x130] += size (0x8132b7 / 0x8132be). Larger: copies what is
//                   buffered, then the vtable's raw read (+0x10).
//   memory stream   0x814aa0: base ctor 0x812e30 (buffer rdx -> +0x120, size rcx -> +0x110 and +0x118), vtable 0x25ad5d0
//                   whose raw read (+0x10) is 0x814b00 `xor eax,eax; ret`: the whole data is the buffer.
// Each fact is checked in the loaded image before the hooks go in; any mismatch: nothing is installed.
#include "loader/demonsSoulsNativeTex.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Loader::NativeTex {

namespace {

// ---- static facts (01.007.000) ----
constexpr uint64_t StreamRead = 0x813130;
constexpr uint64_t MemoryStreamVtable = 0x25ad5d0, MemoryStreamRawRead = 0x814b00;
constexpr uint64_t FileSiteCall = 0x8fed17, PoolSiteCall = 0x8fef15;
// Stream fields (InputStream::Read).
constexpr uint64_t StreamPos = 0x108, StreamEnd = 0x110, StreamBuffer = 0x120, StreamTotal = 0x130;
constexpr uint64_t CaveOffset = 0x8060000, CaveSize = 0x4000; // (Idle 0x8000000, SceneGuard 0x8008000, Camera 0x8020000,
                                                              // NativeMem 0x8040000, GpuPages 0x8080000)

struct Window {
	uint64_t    offset;
	const char* hex;
};
// Exact bytes. The call rel32s inside are part of them: 0x8fed17 and 0x8fef15 are `call 0x813130`.
constexpr std::array<Window, 12> Exact {{
    // 0x8fece7..0x8fed23: Alloc(r13, GPU context, 64 KiB) -> [rbx+0x98]; Read(r15, rax, r13); add [rbx+0x27c],r13d
    {0x8fece7, "488bb558ffffff4989dc4c89efba0000010031c94981c498000000e8e9afefff4989c64c89ff4889c649890424"
               "4c89eae81444f1ff4401ab7c020000"},
    {0x8fec54, "4c89ff4c89e6e9b5000000"},                                   // streamable: rdi = stream, rsi = r12; jmp 0x8fed14
    {0x8feaed, "4d89c4"},                                                   // mov r12,r8 (the CTexture)
    {0x8feb16, "4c89e3"},                                                   // mov rbx,r12
    {0x8fedf4, "4989ff"},                                                   // 0x8fedc0: mov r15,rdi (the pool texture)
    {0x8fef02, "89d8488bbd10fcffff4c89f2480faff04c01e6e81642f1ff"},          // Read(stream, pool + layer * slice, size)
    {0x8fedc0, "554889e54157415641554154534881ecd8030000"},                 // 0x8fedc0 prologue (r15 callee-saved)
    {StreamRead, "554889e54157415641554154534883ec18"},                     // Read prologue
    {0x813141, "4889f0488bb708010000488b9f100100004989d44989d74989fe4829f34929dc0f862e010000"}, // avail; size <= avail
    {0x813295, "31db4d89fceb06488b45d031f64903b6200100004801d84c89e24889c7e8490a66014d01a6080100004d01be30010000"},
    {MemoryStreamRawRead, "31c0c3"},                                        // memory stream raw read: 0
    {0x814ab2, "488d05178bd901498906"},                                     // lea rax,[0x25ad5d0]; mov [r14],rax
}};

struct State {
	uint64_t              base = 0;
	std::vector<std::pair<uint64_t, uint64_t>> segments;
	std::atomic<bool>     installed {false};
	std::atomic<Consumer> consumer {nullptr};
	std::atomic<uint64_t> calls[static_cast<uint32_t>(Site::Count)] {};
	std::atomic<uint64_t> bytes[static_cast<uint32_t>(Site::Count)] {};
	std::atomic<uint64_t> consumed_calls {0}, consumed_bytes {0}, not_memory_stream {0};
};
State g_state;

using ReadFn = KYTY_SYSV_ABI void (*)(uint64_t stream, void* destination, uint64_t size);

bool InImage(uint64_t address, uint64_t size) {
	for (const auto& s: g_state.segments) {
		if (address >= s.first && address < s.second && size <= s.second - address) return true;
	}
	return false;
}

int HexNibble(char c) {
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	return -1;
}

std::vector<uint8_t> Bytes(const char* hex) {
	std::vector<uint8_t> out;
	int                  high = -1;
	for (const char* p = hex; *p != 0; ++p) {
		const int n = HexNibble(*p);
		if (n < 0) continue;
		if (high < 0) {
			high = n;
		} else {
			out.push_back(static_cast<uint8_t>(high << 4 | n));
			high = -1;
		}
	}
	return out;
}

const char* CheckStatic(uint64_t base, uint64_t* where) {
	for (const auto& w: Exact) {
		const auto bytes = Bytes(w.hex);
		*where           = w.offset;
		if (!InImage(base + w.offset, bytes.size())) return "outside the image";
		if (std::memcmp(reinterpret_cast<const void*>(base + w.offset), bytes.data(), bytes.size()) != 0)
			return "instruction bytes differ";
	}
	// The memory stream's relocated raw-read slot.
	*where = MemoryStreamVtable + 0x10;
	if (!InImage(base + MemoryStreamVtable, 0x18)) return "memory stream vtable outside the image";
	uint64_t slot = 0;
	std::memcpy(&slot, reinterpret_cast<const void*>(base + MemoryStreamVtable + 0x10), sizeof(slot));
	if (slot != base + MemoryStreamRawRead) return "memory stream raw-read slot differs";
	return nullptr;
}

uint64_t Load64(uint64_t address) {
	uint64_t v = 0;
	std::memcpy(&v, reinterpret_cast<const void*>(address), sizeof(v));
	return v;
}

void Store64(uint64_t address, uint64_t v) { std::memcpy(reinterpret_cast<void*>(address), &v, sizeof(v)); }

void OriginalRead(uint64_t stream, uint64_t destination, uint64_t size) {
	reinterpret_cast<ReadFn>(g_state.base + StreamRead)(stream, reinterpret_cast<void*>(destination), size);
}

// The hooked call: (rdi, rsi, rdx) of the replaced `call 0x813130`, rcx = the texture (set by the cave).
void Hooked(Site site, uint64_t stream, void* destination, uint64_t size, uint64_t texture) {
	const auto s = static_cast<uint32_t>(site);
	g_state.calls[s].fetch_add(1, std::memory_order_relaxed);
	g_state.bytes[s].fetch_add(size, std::memory_order_relaxed);
	TextureBytes b;
	b.guest_dst = reinterpret_cast<uint64_t>(destination);
	b.size      = size;
	b.stream    = stream;
	b.texture   = texture;
	b.site      = site;
	// Host-readable only for a whole-in-memory stream that holds all of the bytes (Read's single-memcpy path).
	if (stream != 0 && Load64(stream) == g_state.base + MemoryStreamVtable) {
		const uint64_t pos = Load64(stream + StreamPos), end = Load64(stream + StreamEnd);
		const uint64_t buffer = Load64(stream + StreamBuffer);
		if (buffer != 0 && pos <= end && size <= end - pos) b.host_src = reinterpret_cast<const void*>(buffer + pos);
	}
	if (b.host_src == nullptr) g_state.not_memory_stream.fetch_add(1, std::memory_order_relaxed);
	(void)OnTextureBytes(b.guest_dst, b, size);
}

KYTY_SYSV_ABI void HookFile(uint64_t stream, void* destination, uint64_t size, uint64_t texture) {
	Hooked(Site::TextureFile, stream, destination, size, texture);
}
KYTY_SYSV_ABI void HookPool(uint64_t stream, void* destination, uint64_t size, uint64_t texture) {
	Hooked(Site::PoolLayer, stream, destination, size, texture);
}

// Cave: per site `mov rcx, <texture register>; mov rax, host; jmp rax` (rcx is no argument of Read and is caller-saved;
// the return address is the original call's), and the site's call pointed at it. Null when installed.
const char* InstallHooks(uint64_t base) {
	const uint64_t cave = Libs::LibKernel::Memory::AllocateRuntimeMemory(base + CaveOffset, CaveSize,
	                                                                     Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                     "demons_souls_native_tex", true);
	if (cave != base + CaveOffset) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "no cave memory at its address";
	}
	struct Hook {
		uint64_t site;
		uint8_t  mov_rcx[3];
		uint64_t function;
	};
	const Hook hooks[2] = {
	    {FileSiteCall, {0x48, 0x89, 0xd9}, reinterpret_cast<uint64_t>(&HookFile)}, // mov rcx,rbx
	    {PoolSiteCall, {0x4c, 0x89, 0xf9}, reinterpret_cast<uint64_t>(&HookPool)}, // mov rcx,r15
	};
	std::array<int32_t, 2> rel {};
	for (int i = 0; i < 2; i++) {
		const uint64_t stub = cave + 0x40 * static_cast<uint64_t>(i);
		const int64_t  d    = static_cast<int64_t>(stub) - static_cast<int64_t>(base + hooks[i].site + 5);
		if (d != static_cast<int32_t>(d)) {
			Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
			return "cave out of rel32 reach";
		}
		rel[i] = static_cast<int32_t>(d);
		std::array<uint8_t, 15> code {};
		std::memcpy(code.data(), hooks[i].mov_rcx, 3);
		code[3] = 0x48;
		code[4] = 0xb8; // mov rax, imm64
		std::memcpy(code.data() + 5, &hooks[i].function, 8);
		code[13] = 0xff;
		code[14] = 0xe0; // jmp rax
		std::memcpy(reinterpret_cast<void*>(stub), code.data(), code.size());
	}
	if (!Common::VirtualMemory::FlushInstructionCache(cave, 0x80)) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "instruction cache flush failed";
	}
	for (int i = 0; i < 2; i++) {
		std::array<uint8_t, 5> call {0xe8};
		std::memcpy(call.data() + 1, &rel[i], 4);
		std::memcpy(reinterpret_cast<void*>(base + hooks[i].site), call.data(), call.size());
		if (!Common::VirtualMemory::FlushInstructionCache(base + hooks[i].site, call.size()))
			return "instruction cache flush failed (site)";
	}
	return nullptr;
}

} // namespace

void SetConsumer(Consumer consumer) noexcept { g_state.consumer.store(consumer, std::memory_order_release); }

bool Installed() noexcept { return g_state.installed.load(std::memory_order_acquire); }

bool OnTextureBytes(uint64_t guest_dst, const TextureBytes& src, uint64_t size) noexcept {
	const Consumer consumer = g_state.consumer.load(std::memory_order_acquire);
	if (consumer != nullptr && src.host_src != nullptr && size != 0 && src.stream != 0 && consumer(src)) {
		// As InputStream::Read's in-buffer path (0x8132b7 / 0x8132be), without the copy.
		Store64(src.stream + StreamPos, Load64(src.stream + StreamPos) + size);
		Store64(src.stream + StreamTotal, Load64(src.stream + StreamTotal) + size);
		g_state.consumed_calls.fetch_add(1, std::memory_order_relaxed);
		g_state.consumed_bytes.fetch_add(size, std::memory_order_relaxed);
		return true;
	}
	// The original write: the game's own Read, the call the hook replaced.
	OriginalRead(src.stream, guest_dst, size);
	return false;
}

Stats GetStats() noexcept {
	Stats s {};
	for (uint32_t i = 0; i < static_cast<uint32_t>(Site::Count); i++) {
		s.calls[i] = g_state.calls[i].load(std::memory_order_relaxed);
		s.bytes[i] = g_state.bytes[i].load(std::memory_order_relaxed);
	}
	s.consumed_calls    = g_state.consumed_calls.load(std::memory_order_relaxed);
	s.consumed_bytes    = g_state.consumed_bytes.load(std::memory_order_relaxed);
	s.not_memory_stream = g_state.not_memory_stream.load(std::memory_order_relaxed);
	return s;
}

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (!Requested()) return;
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	const auto* elf = program->elf.get();
	if (elf == nullptr || elf->GetEhdr() == nullptr || elf->GetPhdr() == nullptr) return;
	g_state.base     = program->base_vaddr;
	const auto* phdr = elf->GetPhdr();
	for (size_t i = 0; i < elf->GetEhdr()->e_phnum; i++) {
		if (phdr[i].p_type != PT_LOAD && phdr[i].p_type != PT_OS_RELRO) continue;
		const uint64_t begin = g_state.base + phdr[i].p_vaddr, size = phdr[i].p_memsz;
		if (size == 0 || !Libs::Graphics::HostMemoryRangeIsReadable(begin, size)) continue;
		g_state.segments.emplace_back(begin, begin + size);
	}
	std::string version;
	(void)SystemContentParamSfoGetString("APP_VER", &version);
	uint64_t    where  = 0;
	const char* failed = CheckStatic(g_state.base, &where);
	if (failed == nullptr) failed = InstallHooks(g_state.base);
	if (failed != nullptr) {
		std::printf("[native-tex] KYTY_NATIVE_TEX=1 NOT applied: eboot+0x%llx: %s (version %s; checked for 01.007.000); "
		            "textures load as before\n",
		            static_cast<unsigned long long>(where), failed, version.c_str());
		std::fflush(stdout);
		return;
	}
	g_state.installed.store(true, std::memory_order_release);
	std::printf("[native-tex] KYTY_NATIVE_TEX=1: texture loads (eboot+0x%llx FTextureFactory, eboot+0x%llx pool layer) go "
	            "through the host (version %s); %s\n",
	            static_cast<unsigned long long>(FileSiteCall), static_cast<unsigned long long>(PoolSiteCall), version.c_str(),
	            g_state.consumer.load() != nullptr ? "an upload consumer is set"
	                                               : "no upload consumer yet: the original copy runs (no change)");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

} // namespace Loader::NativeTex
