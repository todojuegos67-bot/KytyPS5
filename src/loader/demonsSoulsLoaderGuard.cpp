// Demon's Souls' resource loader (ResourceFactory's loading thread, 01.005.000 eboot+0x88fbc0) takes the first request
// of its priority lists (list nodes: the request at + 0, next at + 0x10) and hands it to the request handling. A request
// withdrawn by another thread leaves its node in two steps: its request cleared first ([node] = 0), then the node
// unlinked and linked to itself. The loader reads the lists without the lock of that removal: a node read in between
// went to the handling with no request, which crashed on it (eboot+0x895822, cmp dword [rdi + 0x5c], 0 with rdi 0).
// Two players of the European 01.005.000 crashed there within minutes of play, one of them also at start-up (issues
// #185, #204); neither the regions' languages nor the job workers' spin settings changed it, and 01.007.000 has the
// same code. Here a node without a request makes the loader wait until it is unlinked (or has a request again; a
// second at most), then scan its lists again from the start.
#include "loader/demonsSoulsLoaderGuard.h"

#include "common/abi.h"
#include "common/guardedCopy.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>

namespace Loader::DemonsSoulsLoaderGuard {
namespace {
constexpr uint64_t PageSize = 0x4000, CaveOffset = 0x8000000 + 6 * PageSize;

// 01.005.000 eboot+0x8906b0: mov rax, [rbx + 0xe0]; cmp rax, [rbp - 0x2250]; je <no request>; then the node taken:
// mov r12, [rax]; cmp [rip + d], r12; je +0x75; mov rax, [rip + d]; mov [rip + d], r12; cmp byte [rax], 0; je +0x62.
constexpr const char* Take = "48 8b 83 e0 00 00 00 48 3b 85 b0 dd ff ff 0f 84 ?? ?? ?? ?? 4c 8b 20 4c 39 25 ?? ?? ?? ?? 74 75 "
                             "48 8b 05 ?? ?? ?? ?? 4c 89 25 ?? ?? ?? ?? 80 38 00 74 62";
constexpr uint64_t SiteOffset = 20; // mov r12, [rax] (3 bytes) and the cmp of the last request (7) move to the stub
constexpr uint32_t SiteSize   = 10;
// The scan's start, 0x3a1 bytes before the site: mov rax, [rbx + d] (the first list's first node); mov [rbp + d], r12d;
// then at the scan's entry cmp rax, [rbp + d]; jne <site> (the list not empty).
constexpr uint64_t ScanDistance = 0x3a1;

uint64_t              g_base = 0;
std::atomic<uint32_t> g_waits {0};

// A node whose request was cleared: wait until it is unlinked (its next is itself) or has a request again, at most a
// second; the loader then scans its lists again.
void KYTY_SYSV_ABI Withdrawn(uint64_t node, uint64_t loader) {
	const auto begin = std::chrono::steady_clock::now();
	for (uint32_t round = 0;; ++round) {
		uint64_t request = 0, next = 0;
		if (Common::GuardedCopy(&request, reinterpret_cast<const void*>(node), sizeof(request)) != 0 ||
		    Common::GuardedCopy(&next, reinterpret_cast<const void*>(node + 0x10), sizeof(next)) != 0 || request != 0 ||
		    next == node || std::chrono::steady_clock::now() - begin >= std::chrono::seconds(1))
			break;
		if (round < 64) std::this_thread::yield();
		else std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
	const auto count = g_waits.fetch_add(1, std::memory_order_relaxed) + 1;
	if (count <= 8 || (count & (count - 1)) == 0)
		std::printf("Demon's Souls loader guard: a request being withdrawn waited out (node 0x%llx of the loader 0x%llx, "
		            "%lld us; %u so far), the lists scanned again\n",
		            static_cast<unsigned long long>(node), static_cast<unsigned long long>(loader),
		            static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                       std::chrono::steady_clock::now() - begin)
		                                       .count()),
		            count);
}

// The scan's start before `site` as 01.005.000 has it; its first list's displacement from the loader (rbx), or 0.
uint32_t ScanStart(uint64_t site, uint64_t& scan) {
	scan            = site - ScanDistance;
	const auto* at  = reinterpret_cast<const uint8_t*>(scan - 14);
	constexpr uint8_t load[] {0x48, 0x8b, 0x83}, store[] {0x44, 0x89, 0xa5}, compare[] {0x48, 0x3b, 0x85}, jne[] {0x0f, 0x85};
	if (std::memcmp(at, load, 3) != 0 || std::memcmp(at + 7, store, 3) != 0 || std::memcmp(at + 14, compare, 3) != 0 ||
	    std::memcmp(at + 21, jne, 2) != 0)
		return 0;
	int32_t to = 0;
	std::memcpy(&to, at + 23, sizeof(to));
	if (scan + 13 + static_cast<int64_t>(to) != site) return 0;
	uint32_t list = 0;
	std::memcpy(&list, at + 3, sizeof(list));
	return list;
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size > CaveOffset || program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	const auto take = GuestCode::FindUnique(*program, GuestCode::Pattern(Take));
	if (!take) return; // (not 01.005.000)
	const auto site = *take + SiteOffset;
	uint64_t   scan = 0;
	const auto list = ScanStart(site, scan);
	if (list == 0) {
		std::printf("Demon's Souls loader guard: the resource loader's scan is not of a known build; retaining its guest code\n");
		return;
	}
	int32_t last = 0;
	std::memcpy(&last, reinterpret_cast<const void*>(site + 6), sizeof(last));
	const uint64_t last_request = site + SiteSize + static_cast<int64_t>(last);
	const auto     requested    = program->base_vaddr + CaveOffset;
	const auto     allocated    = Libs::LibKernel::Memory::AllocateRuntimeMemory(
        requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_loader_guard", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	g_base = program->base_vaddr;
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize, reinterpret_cast<void*>(allocated));
	constexpr uint8_t save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	Xbyak::Label      withdrawn;
	// The node's request; with one, the moved compare of the last request (its address now absolute) and back. (rax, the
	// node, is read no more after it: the code after the site loads it again before a use, or calls first.)
	c.mov(r12, ptr[rax]);
	c.test(r12, r12);
	c.jz(withdrawn);
	c.mov(rax, last_request);
	c.cmp(ptr[rax], r12);
	c.jmp(reinterpret_cast<const void*>(site + SiteSize));
	// None: waited out (registers and FP state kept; below the red zone, on a realigned stack), then the scan from its
	// start, as the loader's loop enters it (rax: the first list's first node; the scan compares before it branches).
	c.L(withdrawn);
	c.lea(rsp, ptr[rsp - 128]);
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
		c.push(reg);
	c.mov(rdi, rax);
	c.mov(rsi, rbx);
	c.mov(rbx, rsp);
	c.and_(rsp, -16);
	c.sub(rsp, 512);
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, reinterpret_cast<uint64_t>(&Withdrawn));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.mov(rsp, rbx);
	for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.lea(rsp, ptr[rsp + 128]);
	c.mov(rax, ptr[rbx + list]);
	c.jmp(reinterpret_cast<const void*>(scan));
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
	code[0]                 = 0xe9;
	const auto displacement = static_cast<int32_t>(allocated - (site + 5));
	std::memcpy(code + 1, &displacement, sizeof(displacement));
	std::memcpy(reinterpret_cast<void*>(site), code, sizeof(code));
	Common::VirtualMemory::FlushInstructionCache(site, sizeof(code));
	std::printf("Demon's Souls loader guard: installed (the resource loader at eboot+0x%llx waits out a request being "
	            "withdrawn)\n",
	            static_cast<unsigned long long>(site - program->base_vaddr));
#endif
}

} // namespace Loader::DemonsSoulsLoaderGuard
