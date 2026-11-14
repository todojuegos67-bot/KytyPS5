// KYTY_ENGINE_TEX_REGISTRY: see demonsSoulsEngineTex.h. Static facts (01.007.000 eboot, ELF vaddrs; read instruction by
// instruction, lab/static/facts/ds-engine-tex-registry.txt, all checked against the eboot):
//   CTexture::Create 0x8ffed0(edi w, esi h, edx depth, ecx slices, r8d mips, r9d EFormat, stack: usage, ..., params)
//     0x9000fd..0x900120: [tex+0x264] = w, +0x268 = h, +0x26c = depth, +0x270 = slices, +0x274 = mips
//     0x900a54/0x900a5b: the T# at +0xbc built by 0x1d19a80 (texture path); 0x900afa: [tex+0x98] = the allocation
//     single epilogue 0x900b4f: mov rax,rbx; add rsp,0x138; pop rbx,r12..r15,rbp; ret (no other ret, no tail jump)
//   CTexture base ctor 0xaa3f60: +0x60 = EFormat, +0x61 = flags (&3 == 1: cube), +0x66 / +0x68 = 0xffff
//   CTexture::FTextureFactory 0x8fe750 (from 0x9012ac)
//     0x8feae1..0x8feafa: Read(stream, tex+0xbc, 0x20): the T# from the file
//     0x8feb3b..0x8feb95: the engine decodes it: +0x264 = width, +0x268 = height, +0x26c = depth (1 for array types
//       11/12/13/15), +0x274 = LAST_LEVEL + 1 (1 for MSAA)
//     streamed (chunk count >= 2, 0x8febac): 0x8fec0f/0x8fec12 run size = [chunks_end-0x40] + [chunks_end-0x44] (the
//       last 0x48-byte chunk's offset + size), 0x8fec1d reserve, 0x8fec3c [tex+0x98] = the run
//     loaded whole: 0x8fed02 Alloc(r13) -> [tex+0x98]; 0x8fed1c [tex+0x27c] += r13d; 0x8fed8e [tex+0x278] = [tex+0x27c]
//     (zeroed at 0x8fe88b): the bytes the engine allocated and loaded
//     0x8fed56/0x8fed5c: T# base = the allocation (0 for a TextureArrayManager layer: those live in the pool's texture)
//     single epilogue 0x8fed9d: add rsp,0x98; pop rbx,r12..r15,rbp; ret; rax = the CTexture (0 on the failure path)
//   CTexture release (0x900bd0 = vtable 0x25afc20 slot 0, 0x900ca0, 0x900f70 from TextureArrayManager 0xac51f5):
//     rsi = [rbx+0x98] (rbx = tex), rdi = [0x2f65f58]; call 0x903460 at 0x900c03 / 0x900cc9 / 0x900fa4
// The stacks: rsp is 16-byte aligned at both epilogue sites (6 pushes + 0x138 / 0x98 after the call's 8) and at the
// three release calls (push rbp, rbx, rax), so the stubs below keep the host call aligned.
#include "loader/demonsSoulsEngineTex.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/engineTexRegistry.h"
#include "graphics/host_gpu/nativeTexturePc.h"
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
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace Loader::DemonsSoulsEngineTex {

namespace {

namespace Reg = Libs::Graphics::EngineTexRegistry;

constexpr uint64_t FactoryEpilogue = 0x8fed9d, FactoryResume = 0x8feda4;
constexpr uint64_t CreateEpilogue = 0x900b4f, CreateResume = 0x900b59;
constexpr uint64_t FreeQueuePush = 0x903460;
constexpr std::array<uint64_t, 3> ReleaseCalls {0x900c03, 0x900cc9, 0x900fa4};
constexpr uint64_t CaveOffset = 0x80c0000, CaveSize = 0x4000;

// CTexture fields.
constexpr uint64_t TexFormat = 0x60, TexFlags = 0x61, TexChunksBegin = 0x70, TexChunksEnd = 0x78,
                   TexAlloc = 0x98, TexTSharp = 0xbc, TexWidth = 0x264, TexHeight = 0x268, TexDepth = 0x26c,
                   TexSlices = 0x270, TexMips = 0x274, TexLoadedBytes = 0x278;
constexpr uint64_t ChunkSize = 0x48;

struct Window {
	uint64_t    offset;
	const char* hex; // 'x' nibbles: any
};
constexpr std::array<Window, 9> Exact {{
    // FTextureFactory: stack-guard check and the epilogue (0x8fed94..0x8fedae)
    {0x8fed94, "488b0a483b4dd07512" "4881c498000000" "5b415c415d415e415f5dc3"},
    // FTextureFactory: the stores this relies on. rax = [rbx+0x78]; ...; esi = [rax-0x44]; rsi += [rax-0x40]
    {0x8fec03, "488b4378" "458b6e04" "4d8b6608" "8b70bc" "480370c0"},
    {0x8fed1c, "4401ab7c020000"},                                    // add [rbx+0x27c],r13d
    {0x8fed53, "4889d8" "898bbc000000" "4488b3c0000000"},             // T# base = the allocation
    {0x8fed88, "8b887c020000" "898878020000"},                       // [rax+0x278] = [rax+0x27c]
    // Create: the field stores (0x9000fd..0x900128) and the epilogue (0x900b3f..0x900b63)
    {0x9000fd, "4189b42464020000" "4189942468020000" "458984246c020000" "4589b42470020000" "83e0fd" "4589bc2474020000"},
    {0x900b3f, "488b05xxxxxxxx" "488b00" "483b45d0" "7515" "4889d8" "4881c438010000" "5b415c415d415e415f5dc3"},
    // release: mov rsi,[rbx+0x98]; test; je; lea rax,[0x2f65f58]; mov rdi,[rax] (the calls are checked as rel32 below)
    {0x900bed, "488bb398000000" "4885f6" "741a" "488d05xxxxxxxx" "488b38"},
    {0x900f8e, "488bb398000000" "4885f6" "741a" "488d05xxxxxxxx" "488b38"},
}};
constexpr Window ReleaseVariant {0x900cb0, "488bb798000000" "4889fb" "4885f6" "741a" "488d05xxxxxxxx" "488b38"};

struct State {
	uint64_t                                   base = 0;
	std::vector<std::pair<uint64_t, uint64_t>> segments;
};
State g_state;
bool  g_registry = false; // KYTY_ENGINE_TEX_REGISTRY=1 (else the hooks serve KYTY_NATIVE_TEXTURE_PC only)
bool  g_hooks    = false;
namespace Pc = Libs::Graphics::NativeTexturePc;
constexpr uint64_t TexArraySlot = 0x65, TexArrayId = 0x66;

bool InImage(uint64_t address, uint64_t size) {
	for (const auto& s: g_state.segments) {
		if (address >= s.first && address < s.second && size <= s.second - address) return true;
	}
	return false;
}

int HexNibble(char c) {
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	if (c == 'x') return 16;
	return -1;
}

// Pairs of nibbles; an 'x' nibble makes the whole byte a wildcard (-1).
std::vector<int> Pattern(const char* hex) {
	std::vector<int> out;
	int              high = -1;
	for (const char* p = hex; *p != 0; ++p) {
		const int n = HexNibble(*p);
		if (n < 0) continue;
		if (high < 0) {
			high = n;
		} else {
			out.push_back(high == 16 || n == 16 ? -1 : (high << 4 | n));
			high = -1;
		}
	}
	return out;
}

bool Matches(uint64_t address, const char* hex) {
	const auto pattern = Pattern(hex);
	if (!InImage(address, pattern.size())) return false;
	const auto* code = reinterpret_cast<const uint8_t*>(address);
	for (size_t i = 0; i < pattern.size(); i++) {
		if (pattern[i] >= 0 && code[i] != pattern[i]) return false;
	}
	return true;
}

bool CallsTo(uint64_t site, uint64_t target) {
	if (!InImage(site, 5)) return false;
	const auto* code = reinterpret_cast<const uint8_t*>(site);
	if (code[0] != 0xe8) return false;
	int32_t rel = 0;
	std::memcpy(&rel, code + 1, sizeof(rel));
	return site + 5 + static_cast<int64_t>(rel) == target;
}

const char* CheckStatic(uint64_t base, uint64_t* where) {
	for (const auto& w: Exact) {
		*where = w.offset;
		if (!Matches(base + w.offset, w.hex)) return "instruction bytes differ";
	}
	*where = ReleaseVariant.offset;
	if (!Matches(base + ReleaseVariant.offset, ReleaseVariant.hex)) return "instruction bytes differ";
	for (const uint64_t site: ReleaseCalls) {
		*where = site;
		if (!CallsTo(base + site, base + FreeQueuePush)) return "not a call to the free queue";
	}
	// The stack-guard failure calls after both epilogues (the only other way out of the two functions).
	*where = 0x8fedaf;
	if (!CallsTo(base + 0x8fedaf, base + 0x1e73c40)) return "stack-guard call differs";
	*where = 0x900b64;
	if (!CallsTo(base + 0x900b64, base + 0x1e73c40)) return "stack-guard call differs";
	return nullptr;
}

template <typename T>
T Load(uint64_t address) {
	T v {};
	std::memcpy(&v, reinterpret_cast<const void*>(address), sizeof(T));
	return v;
}

uint64_t TSharpBase(const uint32_t t[8]) {
	return ((t[0] | (static_cast<uint64_t>(t[1]) << 32u)) & 0xFFFFFFFFFFull) << 8u;
}

void Common(uint64_t tex, Reg::Entry& e) {
	std::memcpy(e.tsharp, reinterpret_cast<const void*>(tex + TexTSharp), sizeof(e.tsharp));
	e.base          = TSharpBase(e.tsharp);
	e.owner         = tex;
	e.width         = Load<uint32_t>(tex + TexWidth);
	e.height        = Load<uint32_t>(tex + TexHeight);
	e.depth         = Load<uint32_t>(tex + TexDepth);
	e.mips          = Load<uint32_t>(tex + TexMips);
	e.engine_format = Load<uint8_t>(tex + TexFormat);
	e.flags         = Load<uint8_t>(tex + TexFlags);
}

// FTextureFactory's return (rax).
KYTY_SYSV_ABI void OnFactory(uint64_t tex) {
	if (tex == 0) return;
	Reg::Entry e {};
	Common(tex, e);
	const uint64_t alloc = Load<uint64_t>(tex + TexAlloc);
	// A TextureArrayManager layer (array id at +0x66 != 0xffff) has no memory of its own: its T# base is 0.
	if (e.base == 0 || alloc == 0) return;
	const uint64_t begin = Load<uint64_t>(tex + TexChunksBegin), end = Load<uint64_t>(tex + TexChunksEnd);
	if (end > begin && (end - begin) / ChunkSize >= 2) {
		e.size   = Load<uint64_t>(end - ChunkSize + 8) + Load<uint32_t>(end - ChunkSize + 4);
		e.source = Reg::Source::FactoryStreamed;
	} else {
		e.size   = Load<uint32_t>(tex + TexLoadedBytes);
		e.source = Reg::Source::Factory;
	}
	if (Pc::Enabled()) {
		Pc::Texture t;
		t.owner  = tex;
		t.base   = alloc; // (= the T# base: 0x8fed56 stores alloc >> 8)
		t.size   = e.size;
		t.width  = e.width;
		t.height = e.height;
		t.depth  = e.depth;
		t.mips   = e.mips;
		t.format = e.engine_format;
		t.kind   = e.source == Reg::Source::FactoryStreamed ? Pc::Kind::Streamed : Pc::Kind::Loaded;
		// AddTexture 0xac4b9c: [+0x65] = 1 - an array texture loaded into its own memory, copied into the pool later.
		t.slice_copy_candidate = Load<uint16_t>(tex + TexArrayId) != 0xffff && Load<uint8_t>(tex + TexArraySlot) == 1;
		Pc::OnTexture(t);
	}
	if (!g_registry) return;
	// The layers of the engine's own T#: DEPTH (dword 4, bits 0..12) is the last layer of an array type (12 1D array,
	// 13 2D array, 15 MSAA array); 1 otherwise (the engine records no slice count for a loaded texture: +0x26c is 1).
	const uint32_t type = e.tsharp[3] >> 28u;
	e.layers            = type == 12 || type == 13 || type == 15 ? (e.tsharp[4] & 0x1fffu) + 1u : 1u;
	Reg::Insert(e);
}

// CTexture::Create's return (rbx).
KYTY_SYSV_ABI void OnCreate(uint64_t tex) {
	if (tex == 0) return;
	Reg::Entry e {};
	Common(tex, e);
	if (e.base == 0) return;
	const uint32_t slices = Load<uint32_t>(tex + TexSlices);
	e.layers              = (e.flags & 3u) == 1u ? slices * 6u : slices;
	e.size                = 0; // the declaration (w, h, slices, mips) bounds it, not a byte count
	e.source              = Reg::Source::Create;
	if (Pc::Enabled()) {
		Pc::Texture t;
		t.owner  = tex;
		t.base   = e.base;
		t.width  = e.width;
		t.height = e.height;
		t.depth  = e.depth;
		t.layers = e.layers;
		t.mips   = e.mips;
		t.format = e.engine_format;
		t.kind   = Pc::Kind::Created;
		Pc::OnTexture(t);
	}
	if (!g_registry) return;
	Reg::Insert(e);
}

// Before the engine's deferred free of [tex+0x98].
KYTY_SYSV_ABI void OnRelease(uint64_t tex, uint64_t pointer) {
	Pc::OnRelease(tex, pointer);
	if (g_registry) Reg::Release(tex, pointer);
}

void Put(std::vector<uint8_t>& code, std::initializer_list<uint8_t> bytes) { code.insert(code.end(), bytes); }
void Put64(std::vector<uint8_t>& code, uint64_t v) {
	for (int i = 0; i < 8; i++) code.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
// jmp rel32 to `target` from the end of the code (cave address `at`). False when out of reach.
bool PutJmp(std::vector<uint8_t>& code, uint64_t at, uint64_t target) {
	const int64_t d = static_cast<int64_t>(target) - static_cast<int64_t>(at + code.size() + 5);
	if (d != static_cast<int32_t>(d)) return false;
	code.push_back(0xe9);
	const auto rel = static_cast<int32_t>(d);
	for (int i = 0; i < 4; i++) code.push_back(static_cast<uint8_t>(static_cast<uint32_t>(rel) >> (8 * i)));
	return true;
}
bool Rel32(uint64_t from_next, uint64_t target, int32_t* rel) {
	const int64_t d = static_cast<int64_t>(target) - static_cast<int64_t>(from_next);
	if (d != static_cast<int32_t>(d)) return false;
	*rel = static_cast<int32_t>(d);
	return true;
}

const char* InstallHooks(uint64_t base) {
	const uint64_t cave = Libs::LibKernel::Memory::AllocateRuntimeMemory(base + CaveOffset, CaveSize,
	                                                                     Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                     "demons_souls_engine_tex", true);
	if (cave != base + CaveOffset) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "no cave memory at its address";
	}
	const uint64_t factory_stub = cave, create_stub = cave + 0x40, release_stub = cave + 0x80;
	std::vector<uint8_t> f, c, r;
	bool                 ok = true;
	// FactoryEpilogue (rsp aligned): push rax; sub rsp,8; mov rdi,rax; call OnFactory; add rsp,8; pop rax;
	// add rsp,0x98; jmp FactoryResume (pop rbx ... ret).
	Put(f, {0x50, 0x48, 0x83, 0xec, 0x08, 0x48, 0x89, 0xc7, 0x48, 0xb8});
	Put64(f, reinterpret_cast<uint64_t>(&OnFactory));
	Put(f, {0xff, 0xd0, 0x48, 0x83, 0xc4, 0x08, 0x58, 0x48, 0x81, 0xc4, 0x98, 0x00, 0x00, 0x00});
	ok = ok && PutJmp(f, factory_stub, base + FactoryResume);
	// CreateEpilogue (rsp aligned): mov rdi,rbx; call OnCreate; mov rax,rbx; add rsp,0x138; jmp CreateResume.
	Put(c, {0x48, 0x89, 0xdf, 0x48, 0xb8});
	Put64(c, reinterpret_cast<uint64_t>(&OnCreate));
	Put(c, {0xff, 0xd0, 0x48, 0x89, 0xd8, 0x48, 0x81, 0xc4, 0x38, 0x01, 0x00, 0x00});
	ok = ok && PutJmp(c, create_stub, base + CreateResume);
	// Release calls (entered by call: rsp = 8 mod 16): push rdi,rsi,rdx,rcx; sub rsp,8; mov rdi,rbx (tex; rsi = the
	// pointer); call OnRelease; add rsp,8; pop rcx,rdx,rsi,rdi; jmp 0x903460 (returns to the original call's caller).
	Put(r, {0x57, 0x56, 0x52, 0x51, 0x48, 0x83, 0xec, 0x08, 0x48, 0x89, 0xdf, 0x48, 0xb8});
	Put64(r, reinterpret_cast<uint64_t>(&OnRelease));
	Put(r, {0xff, 0xd0, 0x48, 0x83, 0xc4, 0x08, 0x59, 0x5a, 0x5e, 0x5f});
	ok = ok && PutJmp(r, release_stub, base + FreeQueuePush);
	if (!ok || f.size() > 0x40 || c.size() > 0x40 || r.size() > 0x40) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "cave out of rel32 reach";
	}
	// The sites: jmp rel32 + nops at the epilogues; the release calls pointed at the release stub.
	int32_t f_rel = 0, c_rel = 0;
	std::array<int32_t, 3> r_rel {};
	ok = Rel32(base + FactoryEpilogue + 5, factory_stub, &f_rel) && Rel32(base + CreateEpilogue + 5, create_stub, &c_rel);
	for (size_t i = 0; i < ReleaseCalls.size(); i++) ok = ok && Rel32(base + ReleaseCalls[i] + 5, release_stub, &r_rel[i]);
	if (!ok) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "cave out of rel32 reach";
	}
	std::memcpy(reinterpret_cast<void*>(factory_stub), f.data(), f.size());
	std::memcpy(reinterpret_cast<void*>(create_stub), c.data(), c.size());
	std::memcpy(reinterpret_cast<void*>(release_stub), r.data(), r.size());
	if (!Common::VirtualMemory::FlushInstructionCache(cave, 0xc0)) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return "instruction cache flush failed";
	}
	const auto patch = [](uint64_t site, uint8_t opcode, int32_t rel, size_t nops) {
		std::array<uint8_t, 10> code {opcode};
		std::memcpy(code.data() + 1, &rel, sizeof(rel));
		// nop forms: 2 bytes 66 90; 5 bytes 0f 1f 44 00 00 (never executed: the stubs jump past them)
		if (nops == 2) {
			code[5] = 0x66;
			code[6] = 0x90;
		} else if (nops == 5) {
			code[5] = 0x0f;
			code[6] = 0x1f;
			code[7] = 0x44;
			code[8] = 0x00;
			code[9] = 0x00;
		}
		std::memcpy(reinterpret_cast<void*>(site), code.data(), 5 + nops);
		return Common::VirtualMemory::FlushInstructionCache(site, 5 + nops);
	};
	bool flushed = patch(base + FactoryEpilogue, 0xe9, f_rel, 2); // add rsp,0x98 (7 bytes)
	flushed      = patch(base + CreateEpilogue, 0xe9, c_rel, 5) && flushed; // mov rax,rbx; add rsp,0x138 (10 bytes)
	for (size_t i = 0; i < ReleaseCalls.size(); i++) flushed = patch(base + ReleaseCalls[i], 0xe8, r_rel[i], 0) && flushed;
	return flushed ? nullptr : "instruction cache flush failed (site)";
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	const char* v = std::getenv("KYTY_ENGINE_TEX_REGISTRY");
	g_registry    = v != nullptr && v[0] == '1';
	if (!g_registry && !Pc::Requested()) return;
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
		std::printf("[engine-tex] %s NOT applied: eboot+0x%llx: %s (version %s; checked for "
		            "01.007.000); textures are sized as before\n",
		            g_registry ? "KYTY_ENGINE_TEX_REGISTRY=1" : "the CTexture hooks of KYTY_NATIVE_TEXTURE_PC=1",
		            static_cast<unsigned long long>(where), failed, version.c_str());
		std::fflush(stdout);
		return;
	}
	g_hooks = true;
	if (!g_registry) {
		std::printf("[engine-tex] CTexture create/factory/release hooks in for KYTY_NATIVE_TEXTURE_PC=1 (the registry "
		            "itself stays off)\n");
		std::fflush(stdout);
		return;
	}
	Reg::SetEbootBase(g_state.base);
	Reg::SetEnabled(true);
	std::printf("[engine-tex] KYTY_ENGINE_TEX_REGISTRY=1: the engine's textures are registered as it makes them "
	            "(FTextureFactory eboot+0x%llx, CTexture::Create eboot+0x%llx, released at the deferred free) and the "
	            "texture cache sizes images by them (version %s)\n",
	            static_cast<unsigned long long>(FactoryEpilogue), static_cast<unsigned long long>(CreateEpilogue),
	            version.c_str());
	std::fflush(stdout);
#else
	(void)program;
#endif
}

bool HooksInstalled() noexcept { return g_hooks; }

} // namespace Loader::DemonsSoulsEngineTex
