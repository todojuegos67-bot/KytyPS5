// (inmortal1) Read-only diagnostic of enemies that take no damage. eboot 01.007.000 (vaddr = Ghidra address, objdump
// of the text segment):
//   0x32cf20 (apply hit, vtable 0x25814d0 + 0x390): 0x32cf77 call 0x32d090 (damage calc), then
//   0x32cf8a e8 a1 06 00 00  call 0x32d630 (rdi chr, rsi, rdx hit, ecx 0, r8d no-subtract flag).
//   0x32d630 skips the HP subtraction (chr+0x710) when: chr+0x7d8 & 0x40; convar 0x27cbd50 == 1 and virtual +0x120 true;
//   allNoDamage 0x27cc200 != 0; r8b != 0; damage hit+0x16c <= 0 (nothing to take); chr+0x580 & 0x40 (dword 0x57e bit 22);
//   virtual +0x400 (0x336500: 0x57e bits 23 / 16 / 17 (unless (*(chr+0x38))->+0x40 ->+0x170 == 1), 0x582 & 0x10,
//   else (*(chr+0x38))->+0x40 ->+0xb7).
// The call at 0x32cf8a goes to a cave that reads those fields (registers, flags and FP state kept) and jumps to 0x32d630
// with the call's own return address. Nothing is written to the game. Logged once per (chr, flags), at most 20 lines.
// Player: *(*(0x27cf378) + 0x258) (demonsSoulsInventory.cpp).
#include "loader/demonsSoulsInmortal.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"

#include <xbyak/xbyak.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace Loader::DemonsSoulsInmortal {
namespace {

constexpr uint64_t CavePage = 0x4000, CaveOffset = 0x8000000 + 28 * 0x4000; // (inventory +24, gpu pages +32)
constexpr uint64_t Site = 0x32cf8a, Target = 0x32d630, Params = 0x27cf378, NoDamage = 0x27cc200, Convar120 = 0x27cbd50;

uint64_t         g_base = 0;
std::atomic<int> g_lines {0};
std::mutex       g_mutex;
struct Seen {
	uint64_t chr;
	uint32_t key;
};
Seen g_seen[64] {};
int  g_seen_count = 0;

template <typename T>
T Rd(uint64_t a) {
	T v {};
	std::memcpy(&v, reinterpret_cast<const void*>(a), sizeof(T));
	return v;
}

void KYTY_SYSV_ABI Observe(uint64_t chr, uint64_t hit, uint64_t r8) {
	if (g_lines.load(std::memory_order_relaxed) >= 20 || chr == 0 || hit == 0) return;
	const int32_t damage = Rd<int32_t>(hit + 0x16c);
	const bool    nosub  = (r8 & 0xff) != 0;
	if (damage < 1 && !nosub) return;
	const uint32_t f7d8 = Rd<uint8_t>(chr + 0x7d8) & 0x40, f57e = Rd<uint32_t>(chr + 0x57e), f582 = Rd<uint16_t>(chr + 0x582) & 0x10;
	const uint8_t  all  = Rd<uint8_t>(g_base + NoDamage), c120 = Rd<uint8_t>(g_base + Convar120);
	int            b7 = -1, s170 = -1;
	if (const uint64_t a = Rd<uint64_t>(chr + 0x38); a != 0)
		if (const uint64_t b = Rd<uint64_t>(a + 0x40); b != 0) {
			b7   = Rd<uint8_t>(b + 0xb7);
			s170 = Rd<uint8_t>(b + 0x170);
		}
	const bool b22 = (f57e >> 22) & 1, b23 = (f57e >> 23) & 1, b16 = (f57e >> 16) & 1, b17 = (f57e >> 17) & 1;
	const bool blocked = f7d8 || all || nosub || b22 || b23 || b16 || (b17 && s170 != 1) || f582 || b7 == 1;
	if (!blocked) return;
	uint64_t player = 0;
	if (const uint64_t p = Rd<uint64_t>(g_base + Params); p != 0) player = Rd<uint64_t>(p + 0x258);
	if (chr == player) return;
	const uint32_t key = (f7d8 ? 1u : 0) | (all ? 2u : 0) | (nosub ? 4u : 0) | (b22 ? 8u : 0) | (b23 ? 16u : 0) | (b16 ? 32u : 0) |
	                     (b17 ? 64u : 0) | (f582 ? 128u : 0) | (b7 == 1 ? 256u : 0);
	{
		std::scoped_lock lock(g_mutex);
		for (int i = 0; i < g_seen_count; i++)
			if (g_seen[i].chr == chr && g_seen[i].key == key) return;
		if (g_seen_count < 64) g_seen[g_seen_count++] = {chr, key};
	}
	if (g_lines.fetch_add(1) >= 20) return;
	std::printf("[inmortal] golpe sin daño: chr 0x%llx vida %d/%d daño %d; flags: 0x7d8&0x40=%d, 0x57e=0x%08x (b22=%d b23=%d b16=%d "
	            "b17=%d s170=%d), 0x582&0x10=%d, b7=%d, allNoDamage=%d, nosub(r8)=%d, cvar0x27cbd50=%d\n",
	            static_cast<unsigned long long>(chr), Rd<int32_t>(chr + 0x710), Rd<int32_t>(chr + 0x718), damage, f7d8 ? 1 : 0, f57e,
	            b22, b23, b16, b17, s170, f582 ? 1 : 0, b7, all, nosub ? 1 : 0, c120);
	std::fflush(stdout);
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin") return;
	const uint64_t base = program->base_vaddr;
	// The game's bytes as expected (01.007.000): mov r8d, r13d; xor ecx, ecx; call 0x32d630; and the callee's prologue
	// push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp, 0x38.
	static constexpr uint8_t site_bytes[] {0x45, 0x89, 0xe8, 0x31, 0xc9, 0xe8, 0xa1, 0x06, 0x00, 0x00};
	static constexpr uint8_t prologue[] {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x38};
	if (program->mapped_size < Params + 8 || std::memcmp(reinterpret_cast<const void*>(base + Site - 5), site_bytes, sizeof(site_bytes)) != 0 ||
	    std::memcmp(reinterpret_cast<const void*>(base + Target), prologue, sizeof(prologue)) != 0) {
		std::printf("[inmortal] off (the game's bytes differ from 01.007.000)\n");
		return;
	}
	const uint64_t cave = Libs::LibKernel::Memory::AllocateRuntimeMemory(base + CaveOffset, CavePage, Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                     "demons_souls_inmortal", true);
	const int64_t distance = static_cast<int64_t>(base + Site) - static_cast<int64_t>(cave);
	if (cave == 0 || distance <= -0x70000000LL || distance >= 0x70000000LL) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CavePage);
		std::printf("[inmortal] off (no executable memory near the eboot)\n");
		return;
	}
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(CavePage, reinterpret_cast<void*>(cave));
	constexpr uint8_t   save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	const uint8_t*      entry = c.getCurr();
	c.lea(rsp, ptr[rsp - 128]);
	c.pushfq();
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
		c.push(reg);
	c.mov(rbx, rsp);
	c.and_(rsp, -16);
	c.sub(rsp, 512);
	c.db(save_fp, sizeof(save_fp));
	c.mov(rsi, rdx); // Observe(chr = rdi, hit = rdx, r8)
	c.mov(rdx, r8);
	c.mov(rax, reinterpret_cast<uint64_t>(&Observe));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.mov(rsp, rbx);
	for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.popfq();
	c.lea(rsp, ptr[rsp + 128]);
	c.jmp(reinterpret_cast<const void*>(base + Target), Xbyak::CodeGenerator::T_NEAR); // (the return address is the call site's)
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(cave, c.getSize())) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CavePage);
		std::printf("[inmortal] off (hook code not generated)\n");
		return;
	}
	g_base = base;
	uint8_t       j[5] {0xe8, 0, 0, 0, 0};
	const int32_t rel = static_cast<int32_t>(reinterpret_cast<int64_t>(entry) - static_cast<int64_t>(base + Site + 5));
	std::memcpy(j + 1, &rel, 4);
	std::memcpy(reinterpret_cast<void*>(base + Site), j, 5);
	Common::VirtualMemory::FlushInstructionCache(base + Site, 5);
	std::printf("[inmortal] diagnostico activo: call at eboot+0x%llx (golpe -> 0x%llx), solo lectura\n", static_cast<unsigned long long>(Site),
	            static_cast<unsigned long long>(Target));
#else
	(void)program;
#endif
}

} // namespace Loader::DemonsSoulsInmortal
