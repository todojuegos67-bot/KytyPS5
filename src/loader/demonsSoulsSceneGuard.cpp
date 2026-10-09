// Demon's Souls' scene nodes keep their children in an eastl::vector (node + 0x30); three traversals of that tree
// read node + 0x98 / + 0xa0 first and crashed on a null child in job workers while areas streamed in (01.007.000
// eboot+0x8daff8 10-03 and 10-05, eboot+0xb9d1e1 10-05, eboot+0xc5cabf 10-06; the 10-06 parent held three null
// children that stayed null). What writes them is not known: the child list mutators (AddChild 0x8da6f0,
// RemoveChild 0x8dae50, InsertChild 0x8daee0) never ran concurrently on one parent nor left a null in an
// instrumented 1-1 walk. Each of the three treats a null node as an empty one here (a find returns nothing, a
// visit visits nothing) and the first skips are printed with the registers that locate the parent's vector.
// Three more pass a state bit of a node down its subtree (bits 0x2000, 0x8000 and 0x4000 of node + 8; 01.007.000
// eboot+0xc5a030, +0xc5a2e0 and +0xc5a570, 01.005.000 +0xc39640, +0xc398c0 and +0xc39b20): their loops over the
// children read each child's word at + 8 in place and crashed on the same null children when the world was torn
// down at "save and quit" (eboot+0xc5a3be 10-07, after the first guard had skipped them 32 times). Their loops
// skip a null child here (nothing to pass the bit to).
#include "loader/demonsSoulsSceneGuard.h"

#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsIdle.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>

namespace Loader::DemonsSoulsSceneGuard {
namespace {
constexpr uint64_t PageSize = 0x4000, CaveOffset = 0x8000000 + 2 * PageSize;

// The traversals' first bytes as both builds seen have them (01.007.000 eboot+0x8dafd0, +0xb9d1d0 and +0xc5ca90;
// 01.005.000 +0x8bf620, +0xb7dd20 and +0xc3bf90), up to their reads of node + 0x98 / + 0xa0, with what a build
// moves (displacements, the frame size) left out.
struct Traversal {
	const char* what;
	const char* code;
};
constexpr std::array<Traversal, 3> Traversals {{
    {"find a node by id (returns it or null)",
     "55 48 89 e5 41 57 41 56 41 55 41 54 53 50 89 c8 89 d3 41 89 f6 49 89 ff 48 89 4d d0 83 e0 fd 83 f8 04 0f 84 ?? ?? "
     "?? ?? 4d 8b a7 98 00 00 00 4d 3b a7 a0 00 00 00 0f 84 ?? ?? ?? ?? 41 83 fe ff"},
    {"visit a subtree", "55 48 89 e5 41 57 41 56 41 55 41 54 53 48 83 ec 58 48 8b 87 a0 00 00 00 4c 8b bf 98 00 00 00 89 75 "
                        "d4 48 89 7d c8 48 89 45 b0 49 39 c7 74 60 48 8d 45 10 48 8b 48 28 48 8b 10 4c 8b 60 08 4c 8b 68 10"},
    {"visit a subtree", "55 48 89 e5 41 57 41 56 41 55 41 54 53 48 81 ec ?? ?? ?? ?? 48 8b 05 ?? ?? ?? ?? 89 b5 5c ff ff ff 48 "
                        "8b 00 48 89 45 d0 48 89 bd 50 ff ff ff 4c 8b af 98 00 00 00 4c 8b a7 a0 00 00 00 4d 39 e5"},
}};
// Each begins with push rbp; mov rbp, rsp; push r15: its stub runs them, the jump to the stub replaces them.
constexpr std::array<uint8_t, 6> Prologue {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};

// The children loops of the bit-passing traversals, as both builds have them: the increment and end test
// (add r12, 8; cmp r12, [parent + 0x38]; je with a short or a near displacement), then the child's load
// (mov REG, [r12]) and the read of its word (mov eax, [REG + 8], or 01.005.000's mov ecx, imm32 first).
constexpr std::array<const char*, 2> ChildLoops {"49 83 c4 08 4d 3b ?? 38 74 ?? 4d 8b ?? 24",
                                                 "49 83 c4 08 4d 3b ?? 38 0f 84 ?? ?? ?? ?? 4d 8b ?? 24"};
struct ChildLoop {
	uint64_t increment = 0; // the loop's add r12, 8: a null child continues there
	uint64_t load      = 0; // mov REG, [r12]
	uint32_t reg       = 0; // REG: r8..r15
	uint32_t moved     = 0; // the bytes the jump to the stub replaces: the load and the next instruction
};

std::vector<ChildLoop> FindChildLoops(const Program& program) {
	std::vector<ChildLoop> loops;
	for (const auto* text: ChildLoops) {
		const GuestCode::Pattern pattern(text);
		for (const auto at: GuestCode::Find(program, pattern, 16)) {
			const auto* code = reinterpret_cast<const uint8_t*>(at);
			const auto* load = code + pattern.Size() - 4;
			// cmp r12, [r8..r15 + 0x38]; mov r8..r15 (not r12), [r12]
			const uint32_t reg = (load[2] >> 3u) & 7u;
			if ((code[6] & 0xf8u) != 0x60u || (load[2] & 0xc7u) != 0x04u || reg == 4) continue;
			uint32_t moved = 0;
			if (load[4] == 0x41 && load[5] == 0x8b && load[6] == (0x40u | reg) && load[7] == 0x08) moved = 8; // mov eax, [REG + 8]
			else if (load[4] == 0xb9) moved = 9; // mov ecx, imm32
			else continue;
			loops.push_back({at, at + pattern.Size() - 4, 8 + reg, moved});
		}
	}
	return loops;
}

// An intrusive list unlink in a job worker's loop (01.005.000 eboot+0xd8c1ed, crashed twice in a row on 10-10 at
// +0xd8c1f5 with rcx = rdx = 0): each object of a list takes its first entry out, `entry->prev->next = entry->next;
// if (entry->next) entry->next->prev = entry->prev`. The entry's prev was null (an entry already taken out, both of
// its links cleared, still the object's first): with a null prev only the write through it is skipped (there is no
// previous entry to update) and the loop goes on as the guest wrote it.
//   mov rdx, [rax+0x50]; mov rcx, [rax+0x58]; mov [rcx+0x50], rdx; mov rax, [rax+0x50]; test rax, rax; je; mov [rax+0x58], rcx
constexpr const char* UnlinkCode = "48 8b 50 50 48 8b 48 58 48 89 51 50 48 8b 40 50 48 85 c0 74 ?? 48 89 48 58";
constexpr uint32_t    UnlinkMoved = 8; // mov rcx, [rax+0x58]; mov [rcx+0x50], rdx

uint64_t              g_base = 0;
std::atomic<uint32_t> g_skips {0};
std::atomic<uint32_t> g_child_skips {0};
std::atomic<uint32_t> g_unlink_skips {0};

void KYTY_SYSV_ABI SkippedUnlink(uint64_t offset, uint64_t entry, uint64_t object) {
	if (g_unlink_skips.fetch_add(1, std::memory_order_relaxed) >= 32) return;
	std::printf("Demon's Souls scene guard: an entry with a null prev unlinked at eboot+0x%llx (entry 0x%llx, object 0x%llx)\n",
	            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(entry),
	            static_cast<unsigned long long>(object));
}

void KYTY_SYSV_ABI Skipped(uint64_t offset, uint64_t caller, uint64_t rbx, uint64_t r14) {
	if (g_skips.fetch_add(1, std::memory_order_relaxed) >= 32) return;
	// (At the recursive visits' call sites rbx points at the null element of the parent's vector, r14 at its end.)
	std::printf("Demon's Souls scene guard: a null node skipped by eboot+0x%llx (caller eboot+0x%llx, rbx 0x%llx, r14 "
	            "0x%llx)\n",
	            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(caller - g_base),
	            static_cast<unsigned long long>(rbx), static_cast<unsigned long long>(r14));
}

void KYTY_SYSV_ABI SkippedChild(uint64_t offset, uint64_t element, uint64_t r14, uint64_t r15) {
	if (g_child_skips.fetch_add(1, std::memory_order_relaxed) >= 32) return;
	// (r12 points at the null element of the parent's vector; the parent is r14 in 01.007.000, r15 in 01.005.000.)
	std::printf("Demon's Souls scene guard: a null child skipped at eboot+0x%llx (element 0x%llx, r14 0x%llx, r15 0x%llx)\n",
	            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(element),
	            static_cast<unsigned long long>(r14), static_cast<unsigned long long>(r15));
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size > CaveOffset || program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	// The null node traversals: the three of a known build, or none of them.
	std::array<uint64_t, Traversals.size()> entries {};
	bool                                    traversals = true;
	for (size_t i = 0; i < Traversals.size() && traversals; i++) {
		const auto entry = GuestCode::FindUnique(*program, GuestCode::Pattern(Traversals[i].code));
		if (!entry || std::memcmp(reinterpret_cast<const void*>(*entry), Prologue.data(), Prologue.size()) != 0) {
			std::printf("Demon's Souls scene guard: no single '%s' traversal of a known build found; retaining their guest "
			            "code\n",
			            Traversals[i].what);
			traversals = false;
		} else {
			entries[i] = *entry;
		}
	}
	const auto loops = FindChildLoops(*program);
	if (loops.empty()) std::printf("Demon's Souls scene guard: no children loop of a known build found; retaining their guest code\n");
	// The list unlink: its mov rcx, [rax+0x58] (4 bytes in).
	uint64_t unlink = 0;
	if (const auto found = GuestCode::FindUnique(*program, GuestCode::Pattern(UnlinkCode))) unlink = *found + 4;
	if (!traversals && loops.empty() && unlink == 0) return;
	const auto requested = program->base_vaddr + CaveOffset;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_scene_guard", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	g_base = program->base_vaddr;
	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator c(PageSize, reinterpret_cast<void*>(allocated));
	constexpr uint8_t save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	std::array<const uint8_t*, Traversals.size()> stubs {};
	for (size_t i = 0; i < Traversals.size() && traversals; i++) {
		const auto   entry = entries[i];
		Xbyak::Label skip;
		c.align(16);
		stubs[i] = c.getCurr();
		c.test(rdi, rdi); // (flags are not preserved across a call)
		c.jz(skip);
		c.db(Prologue.data(), Prologue.size());
		c.jmp(reinterpret_cast<const void*>(entry + Prologue.size()));
		// A null node: printed (registers, flags and FP state kept), then nothing found or visited.
		c.L(skip);
		c.pushfq();
		for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11})
			c.push(reg);
		c.mov(rsi, ptr[rsp + 80]); // the return address
		c.mov(rdi, entry - program->base_vaddr);
		c.mov(rdx, rbx);
		c.mov(rcx, r14);
		c.sub(rsp, 520); // entry rsp is 8 mod 16: 80 + 520 bytes align it for FXSAVE
		c.db(save_fp, sizeof(save_fp));
		c.mov(rax, reinterpret_cast<uint64_t>(&Skipped));
		c.call(rax);
		c.fxrstor64(ptr[rsp]);
		c.add(rsp, 520);
		for (const auto& reg: {r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
			c.pop(reg);
		c.popfq();
		c.xor_(eax, eax);
		c.ret();
	}
	// A children loop: the child's load, a null child continues with the next one, else the moved instruction and back.
	std::vector<const uint8_t*> loop_stubs;
	for (const auto& loop: loops) {
		const auto*        load = reinterpret_cast<const uint8_t*>(loop.load);
		const Xbyak::Reg64 child(static_cast<int>(loop.reg));
		Xbyak::Label       null_child;
		c.align(16);
		loop_stubs.push_back(c.getCurr());
		c.db(load, 4); // mov REG, [r12]
		c.test(child, child);
		c.jz(null_child);
		c.db(load + 4, loop.moved - 4);
		c.jmp(reinterpret_cast<const void*>(loop.load + loop.moved));
		// Printed (registers, flags and FP state kept; below the red zone, on a realigned stack), then the next child.
		c.L(null_child);
		c.lea(rsp, ptr[rsp - 128]);
		c.pushfq();
		for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
			c.push(reg);
		c.mov(rbx, rsp);
		c.and_(rsp, -16);
		c.sub(rsp, 512);
		c.db(save_fp, sizeof(save_fp));
		c.mov(rdi, loop.load - program->base_vaddr);
		c.mov(rsi, r12);
		c.mov(rdx, r14);
		c.mov(rcx, r15);
		c.mov(rax, reinterpret_cast<uint64_t>(&SkippedChild));
		c.call(rax);
		c.fxrstor64(ptr[rsp]);
		c.mov(rsp, rbx);
		for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
			c.pop(reg);
		c.popfq();
		c.lea(rsp, ptr[rsp + 128]);
		c.jmp(reinterpret_cast<const void*>(loop.increment));
	}
	// The unlink: prev loaded; a null prev skips the write through it, else the write; then back after both.
	const uint8_t* unlink_stub = nullptr;
	if (unlink != 0) {
		Xbyak::Label null_prev;
		c.align(16);
		unlink_stub = c.getCurr();
		c.mov(rcx, ptr[rax + 0x58]);
		c.test(rcx, rcx); // (flags are dead here: the guest's next test sets them)
		c.jz(null_prev);
		c.mov(ptr[rcx + 0x50], rdx);
		c.jmp(reinterpret_cast<const void*>(unlink + UnlinkMoved));
		// Printed (registers, flags and FP state kept; below the red zone, on a realigned stack), then on.
		c.L(null_prev);
		c.lea(rsp, ptr[rsp - 128]);
		c.pushfq();
		for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, rbx})
			c.push(reg);
		c.mov(rbx, rsp);
		c.and_(rsp, -16);
		c.sub(rsp, 512);
		c.db(save_fp, sizeof(save_fp));
		c.mov(rdi, unlink - program->base_vaddr);
		c.mov(rsi, ptr[rbx + 72]); // the saved rax: the entry
		c.mov(rdx, ptr[rbx]);      // the saved rbx: the object
		c.mov(rax, reinterpret_cast<uint64_t>(&SkippedUnlink));
		c.call(rax);
		c.fxrstor64(ptr[rsp]);
		c.mov(rsp, rbx);
		for (const auto& reg: {rbx, r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
			c.pop(reg);
		c.popfq();
		c.lea(rsp, ptr[rsp + 128]);
		c.jmp(reinterpret_cast<const void*>(unlink + UnlinkMoved));
	}
	c.ready();
	if (Xbyak::GetError() || !Common::VirtualMemory::FlushInstructionCache(allocated, c.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// Installation occurs before module initializers or guest worker threads run.
	const auto patch = [](uint64_t at, const uint8_t* stub, uint32_t size) {
		std::array<uint8_t, 16> code {};
		code.fill(0xcc); // (the moved bytes after the jump are never reached: nothing jumps into them)
		code[0]                 = 0xe9;
		const auto displacement = static_cast<int32_t>(reinterpret_cast<uint64_t>(stub) - (at + 5));
		std::memcpy(code.data() + 1, &displacement, sizeof(displacement));
		std::memcpy(reinterpret_cast<void*>(at), code.data(), size);
		Common::VirtualMemory::FlushInstructionCache(at, size);
	};
	for (size_t i = 0; i < Traversals.size() && traversals; i++)
		patch(entries[i], stubs[i], 5);
	for (size_t i = 0; i < loops.size(); i++)
		patch(loops[i].load, loop_stubs[i], loops[i].moved);
	if (unlink != 0) patch(unlink, unlink_stub, UnlinkMoved);
	if (traversals)
		std::printf("Demon's Souls scene guard: installed (traversals at eboot+0x%llx, 0x%llx and 0x%llx skip a null node)\n",
		            static_cast<unsigned long long>(entries[0] - program->base_vaddr),
		            static_cast<unsigned long long>(entries[1] - program->base_vaddr),
		            static_cast<unsigned long long>(entries[2] - program->base_vaddr));
	for (const auto& loop: loops)
		std::printf("Demon's Souls scene guard: installed (the children loop at eboot+0x%llx skips a null child)\n",
		            static_cast<unsigned long long>(loop.load - program->base_vaddr));
	if (unlink != 0)
		std::printf("Demon's Souls scene guard: installed (the list unlink at eboot+0x%llx skips a null prev)\n",
		            static_cast<unsigned long long>(unlink - program->base_vaddr));
	else
		std::printf("Demon's Souls scene guard: no list unlink of a known build found; retaining its guest code\n");
#endif
}
} // namespace Loader::DemonsSoulsSceneGuard

