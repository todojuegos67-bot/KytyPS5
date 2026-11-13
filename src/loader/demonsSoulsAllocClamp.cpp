// Demon's Souls (01.007.000) takes all of direct memory at start-up in bpe::memory::PlatformPageAllocator::Init
// (eboot+0x80c200; static notes memory-callsites.md):
//   sceKernelAvailableDirectMemorySize(0, dms, 64 KiB) -> base, size                         (call at eboot+0x80c2b3)
//   tracking tables at the top: AllocateDirect + MapDirect of ~2.5 MiB                         (0x80c339 / 0x80c3f0)
//   allocatorMem ("Available To Allocators") = what is left
//   if allocatorMem > 12 GiB: AllocateDirect + MapDirect of allocatorMem - 12 GiB, kept mapped
//                             for good (~1.08 GiB on Kyty's 13824 MiB store)                   (0x80cbf0 / 0x80cca2)
//   AllocateDirect of min(allocatorMem, 12 GiB), mapped 64 KiB at a time on demand (BatchMap)  (0x80cdd3)
// Answering that one AvailableDirectMemorySize call with at most 12 GiB + 2.4 MiB gives allocatorMem < 12 GiB: the
// engine takes its own "not above 12 GiB" path (the one a 12288 MiB store gave it, minus that store's shortfall: the
// on-demand pool is 12 GiB less 64 KiB), no block is mapped for good, and the direct memory above it stays free for what
// is allocated later outside the allocator (MFSR upscaler blocks, AllocateMainDirect; NCA audio heaps, 2 MiB multiples).
// No guest byte is changed: the call is found here and the answer is clamped in the HLE kernel
// (Memory::SetAvailableDirectClamp, keyed on the call's return address).
//
// How the call is found (no hard-coded address, no instruction bytes assumed beyond `e8 rel32` / `ff 25 rel32`):
//   1. the eboot's PLT stubs `jmp qword [rip+disp32]` whose GOT slot holds Kyty's sceKernelAvailableDirectMemorySize,
//      sceKernelGetDirectMemorySize, sceKernelAllocateDirectMemory and sceKernelMapDirectMemory;
//   2. every `call rel32` to those stubs in the executable segments;
//   3. exactly one call to AvailableDirectMemorySize (and no `call qword [rip+slot]` of it), with the allocator
//      start-up's shape around it: a GetDirectMemorySize call in the 0x400 bytes before, and in the 0x1800 bytes after
//      exactly 3 AllocateDirect and 2 MapDirect calls in the order Allocate, Map, Allocate, Map, Allocate.
// Anything else: nothing is changed and the reason is printed.
#include "loader/demonsSoulsAllocClamp.h"

#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace Loader::DemonsSoulsAllocClamp {
namespace {

// The allocator's on-demand budget (physicalMemoryToKeepUnmapped = min(allocatorMem, 0x300000000)).
constexpr uint64_t AllocatorBudget = 0x300000000ull;
// The answer: 12 GiB + 0x270000. mMemorySystemOverhead for it, roundup64K(((((size >> 16) + 63) >> 2) & ~0xF) + 0x26f437)
// = roundup64K(0xc010 + 0x26f437) = 0x280000, so allocatorMem = 0x300270000 - 0x280000 = 12 GiB: below the
// budget whether the engine tests "> 12 GiB" or ">= 12 GiB" (exactly 12 GiB with ">=" would ask for a 0-byte block).
constexpr uint64_t Limit = AllocatorBudget + 0x270000;
// Where 01.007.000 has the call (static notes); only printed, the shape above decides.
constexpr uint64_t AuditedCall = 0x80c2b3;

constexpr uint64_t ShapeBefore = 0x400, ShapeAfter = 0x1800;

enum Api : int { Available = 0, GetSize = 1, Allocate = 2, Map = 3, ApiCount = 4 };
constexpr std::array<const char*, ApiCount> ApiNames {"sceKernelAvailableDirectMemorySize", "sceKernelGetDirectMemorySize",
                                                      "sceKernelAllocateDirectMemory", "sceKernelMapDirectMemory"};

struct Segment {
	uint64_t begin, end;
	bool     exec;
};

int32_t Rel32(const uint8_t* at) {
	int32_t value = 0;
	std::memcpy(&value, at, sizeof(value));
	return value;
}

std::vector<Segment> Segments(const Program& program) {
	std::vector<Segment> segments;
	const auto*          elf = program.elf.get();
	if (elf == nullptr || elf->GetEhdr() == nullptr || elf->GetPhdr() == nullptr) return segments;
	const auto* phdr = elf->GetPhdr();
	for (size_t i = 0; i < elf->GetEhdr()->e_phnum; i++) {
		// The GOT can be in the RELRO segment (mapped like PT_LOAD by the loader); code only in PT_LOAD.
		if (phdr[i].p_type != PT_LOAD && phdr[i].p_type != PT_OS_RELRO) continue;
		const bool     exec = phdr[i].p_type == PT_LOAD && (phdr[i].p_flags & PF_X) != 0;
		const uint64_t size = exec ? phdr[i].p_filesz : phdr[i].p_memsz;
		if (size == 0) continue;
		const uint64_t begin = program.base_vaddr + phdr[i].p_vaddr;
		if (!Libs::Graphics::HostMemoryRangeIsReadable(begin, size)) continue;
		segments.push_back({begin, begin + size, exec});
	}
	return segments;
}

bool SlotReadable(const std::vector<Segment>& segments, uint64_t slot) {
	for (const auto& s: segments) {
		if (slot >= s.begin && slot <= s.end - sizeof(uint64_t) && s.end - s.begin >= sizeof(uint64_t)) return true;
	}
	return false;
}

struct Found {
	std::unordered_map<uint64_t, int>           stubs; // stub address -> Api
	std::array<uint64_t, ApiCount>              stub_count {};
	uint64_t                                    indirect_available_calls = 0;
	std::array<std::vector<uint64_t>, ApiCount> calls; // call instruction addresses
};

Found Scan(const std::vector<Segment>& segments) {
	Found found;
	std::array<uint64_t, ApiCount> functions {
	    reinterpret_cast<uint64_t>(&Libs::LibKernel::Memory::KernelAvailableDirectMemorySize),
	    reinterpret_cast<uint64_t>(&Libs::LibKernel::Memory::KernelGetDirectMemorySize),
	    reinterpret_cast<uint64_t>(&Libs::LibKernel::Memory::KernelAllocateDirectMemory),
	    reinterpret_cast<uint64_t>(&Libs::LibKernel::Memory::KernelMapDirectMemory)};
	std::vector<uint64_t> available_slots;
	// 1. PLT stubs: ff 25 disp32, GOT slot = next + disp32 (filled by RelocateAll).
	for (const auto& s: segments) {
		if (!s.exec || s.end - s.begin < 6) continue;
		const auto* code = reinterpret_cast<const uint8_t*>(s.begin);
		const auto  size = s.end - s.begin;
		for (uint64_t i = 0; i + 6 <= size; i++) {
			if (code[i] != 0xff || code[i + 1] != 0x25) continue;
			const uint64_t slot = s.begin + i + 6 + static_cast<int64_t>(Rel32(code + i + 2));
			if (!SlotReadable(segments, slot)) continue;
			uint64_t value = 0;
			std::memcpy(&value, reinterpret_cast<const void*>(slot), sizeof(value));
			for (int api = 0; api < ApiCount; api++) {
				if (value != functions[api]) continue;
				found.stubs.emplace(s.begin + i, api);
				found.stub_count[api]++;
				if (api == Available) available_slots.push_back(slot);
			}
		}
	}
	// 2. Calls: e8 rel32 to a stub; ff 15 disp32 (call qword [rip+slot]) straight through AvailableDirect's GOT slot.
	for (const auto& s: segments) {
		if (!s.exec || s.end - s.begin < 6) continue;
		const auto* code = reinterpret_cast<const uint8_t*>(s.begin);
		const auto  size = s.end - s.begin;
		for (uint64_t i = 0; i + 5 <= size; i++) {
			if (code[i] == 0xe8) {
				const uint64_t target = s.begin + i + 5 + static_cast<int64_t>(Rel32(code + i + 1));
				if (const auto stub = found.stubs.find(target); stub != found.stubs.end()) {
					found.calls[stub->second].push_back(s.begin + i);
				}
			} else if (code[i] == 0xff && code[i + 1] == 0x15 && i + 6 <= size) {
				const uint64_t slot = s.begin + i + 6 + static_cast<int64_t>(Rel32(code + i + 2));
				for (const auto available_slot: available_slots) {
					if (slot == available_slot) found.indirect_available_calls++;
				}
			}
		}
	}
	return found;
}

std::vector<uint64_t> Between(const std::vector<uint64_t>& calls, uint64_t begin, uint64_t end) {
	std::vector<uint64_t> in;
	for (const auto call: calls) {
		if (call >= begin && call < end) in.push_back(call);
	}
	std::sort(in.begin(), in.end());
	return in;
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	const char* v = std::getenv("KYTY_DS_ALLOC_PATCH");
	if (v != nullptr && v[0] == '0') return; // (11-13) on by default (retail base PS5 gives the game 12 GiB too); =0 turns it off
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin")
		return;
	const auto base     = program->base_vaddr;
	const auto segments = Segments(*program);
	const auto found    = Scan(segments);

	for (int api = 0; api < ApiCount; api++) {
		std::printf("[port] ds-alloc: %s: %llu PLT stub(s), %llu call(s)\n", ApiNames[api],
		            static_cast<unsigned long long>(found.stub_count[api]),
		            static_cast<unsigned long long>(found.calls[api].size()));
	}
	if (found.stub_count[Available] == 0 || found.stub_count[GetSize] == 0 || found.stub_count[Allocate] == 0 ||
	    found.stub_count[Map] == 0) {
		std::printf("[port] ds-alloc: allocator clamp NOT applied (an import stub was not found)\n");
		return;
	}
	if (found.calls[Available].size() != 1 || found.indirect_available_calls != 0) {
		std::printf("[port] ds-alloc: allocator clamp NOT applied (%llu direct and %llu indirect calls of "
		            "sceKernelAvailableDirectMemorySize; exactly one direct call expected)\n",
		            static_cast<unsigned long long>(found.calls[Available].size()),
		            static_cast<unsigned long long>(found.indirect_available_calls));
		return;
	}
	const uint64_t site = found.calls[Available].front();
	if (site < ShapeBefore) {
		std::printf("[port] ds-alloc: allocator clamp NOT applied (call at an unexpected address)\n");
		return;
	}
	const auto sizes  = Between(found.calls[GetSize], site - ShapeBefore, site);
	const auto allocs = Between(found.calls[Allocate], site, site + ShapeAfter);
	const auto maps   = Between(found.calls[Map], site, site + ShapeAfter);
	std::printf("[port] ds-alloc: sceKernelAvailableDirectMemorySize called once, at eboot+0x%llx (01.007.000: 0x%llx, %s)\n",
	            static_cast<unsigned long long>(site - base), static_cast<unsigned long long>(AuditedCall),
	            site - base == AuditedCall ? "same" : "differs: another build?");
	for (const auto call: allocs)
		std::printf("[port] ds-alloc:   AllocateDirect call at eboot+0x%llx\n", static_cast<unsigned long long>(call - base));
	for (const auto call: maps)
		std::printf("[port] ds-alloc:   MapDirect call at eboot+0x%llx\n", static_cast<unsigned long long>(call - base));
	const bool shape = !sizes.empty() && allocs.size() == 3 && maps.size() == 2 && allocs[0] < maps[0] &&
	                   maps[0] < allocs[1] && allocs[1] < maps[1] && maps[1] < allocs[2];
	if (!shape) {
		std::printf("[port] ds-alloc: allocator clamp NOT applied (not the page allocator's start-up: %llu GetDirectMemorySize "
		            "call(s) before, %llu AllocateDirect and %llu MapDirect call(s) after; 1+, 3 and 2 in the order "
		            "A M A M A expected)\n",
		            static_cast<unsigned long long>(sizes.size()), static_cast<unsigned long long>(allocs.size()),
		            static_cast<unsigned long long>(maps.size()));
		return;
	}
	// The return addresses of the start-up's direct-memory calls (call + 5) are printed at run time.
	Libs::LibKernel::Memory::SetAvailableDirectClamp(site + 5, Limit, site - ShapeBefore, allocs[2] + 6, base);
	std::printf("[port] ds-alloc: allocator clamp applied: the call at eboot+0x%llx is told at most 0x%llx (12 GiB + 2.4 MiB) "
	            "of available direct memory; expected next: no AllocateDirect/MapDirect from eboot+0x%llx/0x%llx (the "
	            "block above 12 GiB), an AllocateDirect of 12 GiB (0x300000000) from eboot+0x%llx\n",
	            static_cast<unsigned long long>(site - base), static_cast<unsigned long long>(Limit),
	            static_cast<unsigned long long>(allocs[1] + 5 - base), static_cast<unsigned long long>(maps[1] + 5 - base),
	            static_cast<unsigned long long>(allocs[2] + 5 - base));
	std::fflush(stdout);
#else
	(void)program;
#endif
}

} // namespace Loader::DemonsSoulsAllocClamp
