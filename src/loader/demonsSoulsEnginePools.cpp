// KYTY_ENGINE_POOLS=1 (diagnostics, off by default): Demon's Souls' memory pools listed with what each holds. Every
// block the engine takes from its page allocator goes through one function (01.007.000 eboot+0x801d00, the pool in rdi,
// the size in rsi), which links the block into its pool: the pool keeps the bytes it holds at + 0x48 and its blocks at
// + 0x70. A thunk at that function's entry notes each pool the first time it takes a block, with the code that asked
// (the return addresses up the frame-pointer chain: the game's code keeps rbp); a thread prints the pools by the bytes
// they hold every 60 s. (The engine's own AllocationTracker counts nothing in a retail build.)
#include "loader/demonsSoulsEnginePools.h"

#include "common/abi.h"
#include "common/guardedCopy.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace Loader::DemonsSoulsEnginePools {
namespace {
constexpr uint64_t PageSize = 0x4000, CaveOffset = 0x8000000 + 10 * PageSize;

// 01.007.000 eboot+0x801d00: push rbp; mov rbp, rsp; push r14; push rbx; sub rsp, 0x10; mov r14, [rip + d]; mov rax,
// [r14]; mov [rbp - 0x18], rax; cmp byte [rdi + 9], 1; jne +d; add rsi, 0xffff.
constexpr const char* Take = "55 48 89 e5 41 56 53 48 83 ec 10 4c 8b 35 ?? ?? ?? ?? 49 8b 06 48 89 45 e8 80 7f 09 01 75 ?? "
                             "48 81 c6 ff ff 00 00";
constexpr uint64_t MovedPrologue = 7; // push rbp; mov rbp, rsp; push r14; push rbx
constexpr uint64_t HeldAt = 0x48, BlocksAt = 0x70;
constexpr size_t   Callers = 4;

struct Pool {
	std::atomic<uint64_t>           pool {0};
	std::atomic<bool>               ready {false};
	std::array<uint64_t, Callers>   callers {};
};
std::array<Pool, 4096> g_pools;
uint64_t               g_base = 0;

uint64_t Read(uint64_t address) {
	uint64_t value = 0;
	return Common::GuardedCopy(&value, reinterpret_cast<const void*>(address), sizeof(value)) == 0 ? value : 0;
}

// The bytes of a pool's blocks in this process's working set through the guest view (its block list: the sentinel at
// + 0x50, a block's next at + 8; a block's size at + 0x50, address at + 0x60).
uint64_t Resident(uint64_t pool) {
#if defined(_WIN32)
	constexpr uint64_t Page = 0x1000;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> pages;
	uint64_t                                      resident = 0;
	const uint64_t                                sentinel = pool + 0x50;
	uint64_t                                      node     = Read(sentinel + 8);
	for (uint32_t steps = 0; node != 0 && node != sentinel && steps < 100000; steps++, node = Read(node + 8)) {
		const uint64_t address = Read(node + 0x60), size = Read(node + 0x50);
		for (uint64_t at = 0; at < size; at += Page * 65536) {
			const uint64_t count = std::min<uint64_t>(65536, (size - at) / Page);
			pages.assign(count, {});
			for (uint64_t i = 0; i < count; i++)
				pages[i].VirtualAddress = reinterpret_cast<void*>(address + at + i * Page);
			if (K32QueryWorkingSetEx(GetCurrentProcess(), pages.data(), static_cast<DWORD>(count * sizeof(pages[0]))) == 0)
				break;
			for (const auto& page: pages)
				if (page.VirtualAttributes.Valid != 0) resident += Page;
		}
	}
	return resident;
#else
	(void)pool;
	return 0;
#endif
}

// A pool the wrapper took a block for (`ret`: where it was called from, `frame`: that caller's rbp).
void KYTY_SYSV_ABI OnPool(uint64_t pool, uint64_t ret, uint64_t frame) {
	const size_t start = static_cast<size_t>(((pool >> 4u) * 0x9E3779B97F4A7C15ull) >> 52u);
	for (size_t i = 0; i < g_pools.size(); i++) {
		auto&    entry = g_pools[(start + i) % g_pools.size()];
		uint64_t held  = entry.pool.load(std::memory_order_acquire);
		if (held == pool) return;
		if (held != 0) continue;
		if (!entry.pool.compare_exchange_strong(held, pool, std::memory_order_acq_rel)) {
			if (held == pool) return;
			continue;
		}
		entry.callers[0] = ret;
		for (size_t k = 1; k < Callers && frame != 0; k++) {
			entry.callers[k] = Read(frame + 8);
			frame            = Read(frame);
		}
		entry.ready.store(true, std::memory_order_release);
		return;
	}
}

void Reporter() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(60));
		struct Row {
			uint64_t pool, held, blocks, vtable;
			const Pool* entry;
		};
		std::vector<Row> rows;
		uint64_t         total = 0;
		for (const auto& entry: g_pools) {
			if (!entry.ready.load(std::memory_order_acquire)) continue;
			const uint64_t pool = entry.pool.load(std::memory_order_relaxed);
			const uint64_t held = Read(pool + HeldAt);
			rows.push_back({pool, held, Read(pool + BlocksAt) & 0xffffffffu, Read(pool), &entry});
			total += held;
		}
		std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.held > b.held; });
		std::printf("Engine pools: %zu pools hold %.0f MiB of the page allocator's blocks\n", rows.size(),
		            static_cast<double>(total) / 1048576.0);
		for (size_t i = 0; i < rows.size() && i < 40; i++) {
			const auto& row = rows[i];
			std::printf("  %8.1f MiB (%8.1f MiB resident) %6llu blocks pool 0x%llx vtable eboot+0x%llx callers",
			            static_cast<double>(row.held) / 1048576.0, static_cast<double>(Resident(row.pool)) / 1048576.0,
			            static_cast<unsigned long long>(row.blocks), static_cast<unsigned long long>(row.pool),
			            static_cast<unsigned long long>(row.vtable - g_base));
			for (const auto caller: row.entry->callers)
				if (caller != 0) std::printf(" eboot+0x%llx", static_cast<unsigned long long>(caller - g_base));
			std::printf("\n");
		}
		std::fflush(stdout);
	}
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	const char* value = std::getenv("KYTY_ENGINE_POOLS");
	if (value == nullptr || *value != '1' || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin" || program->mapped_size > CaveOffset ||
	    program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	const auto take = GuestCode::FindUnique(*program, GuestCode::Pattern(Take));
	if (!take) {
		std::printf("Engine pools: the page allocator's wrapper is not of a known build\n");
		return;
	}
	const auto requested = program->base_vaddr + CaveOffset;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_engine_pools", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	g_base = program->base_vaddr;
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize, reinterpret_cast<void*>(allocated));
	// The registers and flags kept around OnPool (rdi, the pool, is its first argument), then the moved prologue and back.
	// Entry RSP is 8 mod 16: 80 bytes saved make it 8 more, 8 more 0 for the call.
	c.pushfq();
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11})
		c.push(reg);
	c.sub(rsp, 8);
	c.mov(rsi, ptr[rsp + 88]); // the return address
	c.mov(rdx, rbp);           // the caller's frame
	c.mov(rax, reinterpret_cast<uint64_t>(&OnPool));
	c.call(rax);
	c.add(rsp, 8);
	for (const auto& reg: {r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.popfq();
	c.push(rbp);
	c.mov(rbp, rsp);
	c.push(r14);
	c.push(rbx);
	c.jmp(reinterpret_cast<const void*>(*take + MovedPrologue), Xbyak::CodeGenerator::T_NEAR);
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(allocated, c.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// jmp thunk; nops over the rest of the moved prologue (installed before guest threads run; nothing jumps into it).
	uint8_t code[MovedPrologue];
	std::memset(code, 0x90, sizeof(code));
	code[0]                 = 0xe9;
	const auto displacement = static_cast<int32_t>(allocated - (*take + 5));
	std::memcpy(code + 1, &displacement, sizeof(displacement));
	std::memcpy(reinterpret_cast<void*>(*take), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(*take, sizeof(code));
	std::thread(Reporter).detach();
	std::printf("Engine pools: the page allocator's wrapper at eboot+0x%llx watched (KYTY_ENGINE_POOLS)\n",
	            static_cast<unsigned long long>(*take - program->base_vaddr));
#else
	(void)program;
#endif
}

} // namespace Loader::DemonsSoulsEnginePools
