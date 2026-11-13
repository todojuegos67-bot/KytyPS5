// KYTY_NATIVE_MEM: see demonsSoulsNativeMem.h. Static facts (01.007.000 eboot, ELF vaddrs; verified instruction by
// instruction, /home/claude/static/native-gpu-resources.md and native-port-plan.md):
//   CRenderDisplay ctor   0x90b271: lea rax,[0x2f0acf8]; mov rsi,r15; mov rdi,[rax]; call 0x91c680
//                                    (RenderTargetManager* in global 0x2f0acf8; InitSharedMemory runs once, here)
//                         0x90b295: call 0x90a8b0 (-> sceVideoOutRegisterBuffers2): after InitSharedMemory
//   InitSharedMemory      0x91c783: lea rsi,"RTSharedAllocator" (0x24a93a6); 0x91c795 call IAllocator ctor 0x8018e0
//                         0x91c79a: lea r13, vtable 0x25afdf0 (RenderTargetArenaAllocator), stored at [obj]
//                         mgr+0x88 = shared arena; 0x91c829 call RequestPages 0x801d00 (size config+0x24, 64 KiB)
//                         0x91c837: arena+0x88 = info+0x60 (VA), arena+0x90 = arena+0xb0 = info+0x58 (size)
//                         0x91c8c1: lea rsi,"RTPersistAllocator" (0x242da5b); 0x91c8d3 ctor; mgr+0xa8 = persist arena
//                         0x91c960 / 0x91c96e: its RequestPages and the same two stores
//   IAllocator ctor       0x8018f8: [rdi+0x10] = name (char*), [rdi+0x18] = EMemoryType (2 = GPUMemory)
//   arena vtable slots    +0xa8 Alloc 0x915990 (linear bump), +0xb0 Free 0x91d520 (ret); Reset 0x915930 only moves
//                         the cursor: the arena's range never changes
//   IAllocator page runs  FindPages 0x801b80: first node [A+0x60], sentinel A+0x50 ([A+0x50] = 0), node[0] = the
//                         PageAllocationInfo itself, next [info+0x10], VA [info+0x60], size [info+0x58]
//                         RequestPages 0x801d45: the insertion that builds that list ([info+0x18] = A+0x50)
//   allocator registry    0x7f8b5a lea rax,[0x2ea0df0]; 0x7f8b70 first free slot of 1..255 (8 bytes each); 0x7f8b8a
//                         stores the IAllocator* there (every allocator, the arenas included: 0x91c7f7)
// Each fact is checked in the loaded image before anything uses it; any mismatch turns the module off.
#include "loader/demonsSoulsNativeMem.h"

#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include <psapi.h>
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <sys/mman.h>
#endif

namespace Loader::DemonsSoulsNativeMem {

std::atomic<int> g_level {0};

namespace {

// ---- static facts (01.007.000) ----
constexpr uint64_t RtManagerGlobal = 0x2f0acf8;
constexpr uint64_t InitSharedMemory = 0x91c680;
constexpr uint64_t ArenaVtable = 0x25afdf0;
constexpr uint64_t ArenaAlloc = 0x915990, ArenaFree = 0x91d520;
constexpr uint64_t SharedName = 0x24a93a6, PersistName = 0x242da5b;
constexpr uint64_t RequestPages = 0x801d00, AllocatorCtor = 0x8018e0;
constexpr uint64_t Registry = 0x2ea0df0;
constexpr uint64_t RegistrySlots = 0x100;
// (The arena sizes the ctor's mode 4 gives, tables 0x24f3e80 / 0x24f8c30: printed beside what was found.)
constexpr uint64_t ExpectedShared = 0x2af40000, ExpectedPersist = 0x38170000;

struct Window {
	uint64_t    offset;
	const char* hex;
};
// Exact bytes (no relocation in any of them).
constexpr std::array<Window, 10> Exact {{
    {0x91c837, "4c8b40604d8987880000004c8b48584d898f900000004d898fb0000000"}, // shared: VA, size, left
    {0x91c96e, "4c8b40604d8986880000004c8b48584d898e900000004d898eb0000000"}, // persist: the same
    {0x8018f8, "48897710895718"},                                         // ctor: name +0x10, type +0x18
    {ArenaFree, "c3"},                                                    // Free: nothing
    {0x915930, "488b8790000000c5f857c0488987b0000000c5f81187a0000000c3"},  // Reset: cursor only
    {0x801b80, "488b47604883c7504839f8750731c0c3488b4010488b004885c074f1488b48604839ce72eb48034858"
               "4839ce73e2ebdf"},                                         // FindPages: the run list
    {0x801d45, "488d4b5048890048894808488b5360488950104889436048"
               "8b5010488942 08ff4370488948 18488b4858 48014b48"},        // RequestPages: the insertion
    {0x7f8b70, "48833cf000740e48ffc64881fe0001000075ed31f689f148"},        // registry: slots 1..255
    {0x7f8b8a, "48891cc8"},                                               // registry: [rax+rcx*8] = allocator
    {0x90b278, "4c89fe488b38"},                                           // mov rsi,r15; mov rdi,[rax]
}};
// rip-relative operands and call targets: {instruction, its length, displacement offset, target}.
struct Rel {
	uint64_t at, length, disp, target;
	uint8_t  opcode0, opcode1, opcode2; // 0 = not checked
};
constexpr std::array<Rel, 10> Rels {{
    {0x90b271, 7, 3, RtManagerGlobal, 0x48, 0x8d, 0x05},
    {0x90b27e, 5, 1, InitSharedMemory, 0xe8, 0, 0},
    {0x91c783, 7, 3, SharedName, 0x48, 0x8d, 0x35},
    {0x91c795, 5, 1, AllocatorCtor, 0xe8, 0, 0},
    {0x91c79a, 7, 3, ArenaVtable, 0x4c, 0x8d, 0x2d},
    {0x91c829, 5, 1, RequestPages, 0xe8, 0, 0},
    {0x91c8c1, 7, 3, PersistName, 0x48, 0x8d, 0x35},
    {0x91c8d3, 5, 1, AllocatorCtor, 0xe8, 0, 0},
    {0x91c960, 5, 1, RequestPages, 0xe8, 0, 0},
    {0x7f8b5a, 7, 3, Registry, 0x48, 0x8d, 0x05},
}};

struct Segment {
	uint64_t begin = 0, end = 0;
};

struct State {
	uint64_t                base = 0;
	std::vector<Segment>    segments;
	std::atomic<bool>       probed {false};
	std::atomic<int>        arenas {0};
	std::array<uint64_t, 2> arena_va {}, arena_size {};
};
State g_state;

bool InImage(uint64_t address, uint64_t size) {
	for (const auto& s: g_state.segments) {
		if (address >= s.begin && address < s.end && size <= s.end - address) return true;
	}
	return false;
}

// Guest memory read without faults: the eboot's own segments directly, the rest through the backing alias (unmapped
// parts read as zeros, which no check below accepts).
// `direct` (the probe, on the game's own thread): mapped guest memory outside the store is read as the game reads it.
// The census thread never reads guest memory directly (an unmap meanwhile would fault).
bool ReadGuest(uint64_t address, void* out, uint64_t size, bool direct = false) {
	if (size == 0 || UINT64_MAX - address < size) return false;
	if (InImage(address, size)) {
		std::memcpy(out, reinterpret_cast<const void*>(address), size);
		return true;
	}
	if (address < 0x10000) return false;
	if (Libs::LibKernel::Memory::TryReadMappedOrZero(address, out, size)) return true;
	if (!direct || !Libs::LibKernel::Memory::IsFullyMapped(address, size) ||
	    !Libs::Graphics::HostMemoryRangeIsReadable(address, size))
		return false;
	std::memcpy(out, reinterpret_cast<const void*>(address), size);
	return true;
}

uint64_t ReadU64(uint64_t address, bool* ok, bool direct = false) {
	uint64_t value = 0;
	if (!ReadGuest(address, &value, sizeof(value), direct)) *ok = false;
	return value;
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
		if (n < 0) continue; // (spaces)
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
	for (const auto& r: Rels) {
		*where = r.at;
		if (!InImage(base + r.at, r.length)) return "outside the image";
		const auto* code = reinterpret_cast<const uint8_t*>(base + r.at);
		if (code[0] != r.opcode0 || (r.opcode1 != 0 && code[1] != r.opcode1) || (r.opcode2 != 0 && code[2] != r.opcode2))
			return "opcode differs";
		int32_t disp = 0;
		std::memcpy(&disp, code + r.disp, sizeof(disp));
		if (r.at + r.length + static_cast<int64_t>(disp) != r.target) return "operand target differs";
	}
	*where = SharedName;
	if (!InImage(base + SharedName, 18) || std::memcmp(reinterpret_cast<const void*>(base + SharedName), "RTSharedAllocator", 18) != 0)
		return "RTSharedAllocator name";
	*where = PersistName;
	if (!InImage(base + PersistName, 19) ||
	    std::memcmp(reinterpret_cast<const void*>(base + PersistName), "RTPersistAllocator", 19) != 0)
		return "RTPersistAllocator name";
	*where = ArenaVtable;
	if (!InImage(base + ArenaVtable, 0xc0)) return "arena vtable outside the image";
	return nullptr;
}

// ---- events ----
constexpr uint64_t RingBits = 17, RingSize = uint64_t {1} << RingBits;
std::atomic<uint64_t> g_ring[RingSize];
std::atomic<uint64_t> g_head {0};
// Level 2 counters: [why] images and bytes kept in video memory.
std::atomic<uint64_t> g_kept_images[2], g_kept_bytes[2];

uint64_t Pack(Event kind, uint64_t vaddr, uint64_t size) {
	const uint64_t pages = std::min<uint64_t>(std::max<uint64_t>((size + 0xfff) >> 12u, 1), (uint64_t {1} << 21u) - 1);
	return (static_cast<uint64_t>(kind) << 61u) | (pages << 40u) | ((vaddr >> 12u) & ((uint64_t {1} << 40u) - 1));
}

// ---- the engine's allocators and their page runs ----
struct Run {
	uint64_t begin, end;
	uint32_t owner; // index into Census::owners
};
struct Owner {
	std::string name;
	uint64_t    object = 0;
	uint32_t    type = 0, slot = 0;
	uint64_t    held = 0; // bytes in its runs
	uint64_t    runs = 0;
	// Since the last report (events, pages of protection changes, write-back bytes).
	uint64_t events[EventKinds] {};
	uint64_t protect_pages = 0, writeback_bytes = 0;
	// Last resident sample.
	uint64_t resident = 0, sampled = 0;
};

struct Census {
	std::vector<Owner> owners; // [0] = memory no engine allocator's run holds
	std::vector<Run>   runs;   // sorted by begin
	uint64_t           tail = 0, dropped = 0;
	uint64_t           totals[EventKinds] {};
	uint64_t           protect_pages = 0, writeback_bytes = 0;
	uint64_t           walk_errors = 0;
};

uint32_t OwnerIndex(Census& c, uint64_t object, uint32_t slot, uint32_t type, const std::string& name) {
	for (uint32_t i = 1; i < c.owners.size(); i++) {
		if (c.owners[i].object == object && c.owners[i].slot == slot && c.owners[i].name == name) return i;
	}
	Owner owner;
	owner.name   = name;
	owner.object = object;
	owner.slot   = slot;
	owner.type   = type;
	c.owners.push_back(owner);
	return static_cast<uint32_t>(c.owners.size() - 1);
}

std::string ReadName(uint64_t pointer) {
	char text[48] {};
	if (pointer == 0 || !ReadGuest(pointer, text, sizeof(text) - 1)) return {};
	std::string name;
	for (const char ch: text) {
		if (ch == 0) break;
		if (ch < 32 || ch > 126) return {};
		name += ch;
	}
	return name;
}

// The registry's allocators and their runs (racy reads of lists the game changes under its own lock: every pointer is
// checked against the list's shape, a walk stops at the first mismatch; a run misattributed for 10 s is the cost).
void Walk(Census& c) {
	const uint64_t base = g_state.base;
	for (auto& owner: c.owners) {
		owner.held = 0;
		owner.runs = 0;
	}
	std::vector<Run> runs;
	for (uint64_t slot = 1; slot < RegistrySlots; slot++) {
		bool           ok        = true;
		const uint64_t allocator = ReadU64(base + Registry + slot * 8, &ok);
		if (!ok || allocator == 0 || (allocator & 7u) != 0) continue;
		uint8_t header[0x88] {};
		if (!ReadGuest(allocator, header, sizeof(header))) continue;
		uint64_t vtable = 0, name_pointer = 0, sentinel_value = 0, first = 0;
		uint32_t type   = 0;
		std::memcpy(&vtable, header, 8);
		std::memcpy(&name_pointer, header + 0x10, 8);
		std::memcpy(&type, header + 0x18, 4);
		std::memcpy(&sentinel_value, header + 0x50, 8);
		std::memcpy(&first, header + 0x60, 8);
		if (!InImage(vtable, 8) || type > 2 || sentinel_value != 0) {
			c.walk_errors++;
			continue;
		}
		std::string name = ReadName(name_pointer);
		if (name.empty()) name = "?";
		const uint32_t owner    = OwnerIndex(c, allocator, static_cast<uint32_t>(slot), type, name);
		const uint64_t sentinel = allocator + 0x50;
		uint64_t       node     = first;
		for (uint32_t steps = 0; node != sentinel && node != 0 && steps < 65536; steps++) {
			uint8_t info[0x68] {};
			if (!ReadGuest(node, info, sizeof(info))) break;
			uint64_t self = 0, next = 0, list = 0, size = 0, va = 0;
			std::memcpy(&self, info, 8);
			std::memcpy(&next, info + 0x10, 8);
			std::memcpy(&list, info + 0x18, 8);
			std::memcpy(&size, info + 0x58, 8);
			std::memcpy(&va, info + 0x60, 8);
			if (self != node || list != sentinel || size == 0 || (va & 0xffffu) != 0 || UINT64_MAX - va < size) {
				c.walk_errors++;
				break;
			}
			runs.push_back({va, va + size, owner});
			c.owners[owner].held += size;
			c.owners[owner].runs++;
			node = next;
		}
	}
	std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.begin < b.begin; });
	c.runs = std::move(runs);
}

uint32_t Classify(const Census& c, uint64_t address) {
	auto it = std::upper_bound(c.runs.begin(), c.runs.end(), address, [](uint64_t a, const Run& r) { return a < r.begin; });
	if (it == c.runs.begin()) return 0;
	--it;
	return address < it->end ? it->owner : 0;
}

void Drain(Census& c) {
	const uint64_t head = g_head.load(std::memory_order_acquire);
	if (head - c.tail > RingSize) {
		c.dropped += head - c.tail - RingSize;
		c.tail = head - RingSize;
	}
	for (; c.tail < head; c.tail++) {
		const uint64_t packed = g_ring[c.tail & (RingSize - 1)].load(std::memory_order_relaxed);
		const auto     kind   = static_cast<uint32_t>(packed >> 61u);
		if (kind >= EventKinds) continue;
		const uint64_t pages   = (packed >> 40u) & ((uint64_t {1} << 21u) - 1);
		const uint64_t address = (packed & ((uint64_t {1} << 40u) - 1)) << 12u;
		auto&          owner   = c.owners[Classify(c, address)];
		owner.events[kind]++;
		c.totals[kind]++;
		if (kind == Protect) {
			owner.protect_pages += pages;
			c.protect_pages += pages;
		} else if (kind == WriteBack) {
			owner.writeback_bytes += pages << 12u;
			c.writeback_bytes += pages << 12u;
		}
	}
}

// One page in each 64 KiB of each run, through the game's view and the backing alias (Windows; Linux: the view).
void SampleResidency(Census& c) {
	constexpr uint64_t Step = 0x10000, MaxSamples = uint64_t {1} << 20u;
	for (auto& owner: c.owners) owner.resident = owner.sampled = 0;
	uint64_t total = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries;
	std::vector<uint32_t>                         owners;
	std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	const auto flush = [&] {
		if (entries.empty()) return;
		if (K32QueryWorkingSetEx(GetCurrentProcess(), entries.data(), static_cast<DWORD>(entries.size() * sizeof(entries[0]))) != 0) {
			for (size_t i = 0; i + 1 < entries.size(); i += 2) {
				const bool resident = (entries[i].VirtualAttributes.Flags & 1u) != 0 ||
				                      (entries[i + 1].VirtualAttributes.Flags & 1u) != 0;
				auto& owner = c.owners[owners[i / 2]];
				owner.sampled += Step;
				if (resident) owner.resident += Step;
			}
		}
		entries.clear();
		owners.clear();
		SwitchToThread();
	};
	for (const auto& run: c.runs) {
		pieces.clear();
		const bool backed = Libs::LibKernel::Memory::TryGetSparseBackingPieces(run.begin, run.end - run.begin, pieces);
		size_t     piece = 0;
		uint64_t   piece_begin = 0;
		for (uint64_t at = run.begin; at < run.end && total < MaxSamples; at += Step, total++) {
			uint64_t alias = 0;
			if (backed) {
				const uint64_t offset = at - run.begin;
				while (piece < pieces.size() && offset >= piece_begin + pieces[piece].second) piece_begin += pieces[piece++].second;
				if (piece < pieces.size() && pieces[piece].first != nullptr)
					alias = reinterpret_cast<uint64_t>(pieces[piece].first) + (offset - piece_begin);
			}
			PSAPI_WORKING_SET_EX_INFORMATION view {}, other {};
			view.VirtualAddress  = reinterpret_cast<PVOID>(at);
			other.VirtualAddress = reinterpret_cast<PVOID>(alias != 0 ? alias & ~uint64_t {0xfff} : at);
			entries.push_back(view);
			entries.push_back(other);
			owners.push_back(run.owner);
			if (entries.size() >= 2048) flush();
		}
	}
	flush();
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	for (const auto& run: c.runs) {
		for (uint64_t at = run.begin; at < run.end && total < MaxSamples; at += Step, total++) {
			unsigned char resident = 0;
			auto&         owner    = c.owners[run.owner];
			owner.sampled += Step;
			if (mincore(reinterpret_cast<void*>(at), 4096, &resident) == 0 && (resident & 1u) != 0) owner.resident += Step;
		}
	}
#else
	(void)total;
#endif
}

constexpr uint64_t MiB = uint64_t {1} << 20u;

void Report(Census& c, double seconds, bool residency) {
	const auto per = [seconds](uint64_t n) { return static_cast<unsigned long long>(static_cast<double>(n) / seconds + 0.5); };
	std::printf("[native-mem] %.0f s: per s faults window %llu write %llu read %llu, protect calls %llu (%llu MiB), "
	            "write-backs %llu (%llu MiB); %zu engine allocators, %zu page runs%s%llu events dropped\n",
	            seconds, per(c.totals[FaultWindow]), per(c.totals[FaultWrite]), per(c.totals[FaultRead]), per(c.totals[Protect]),
	            static_cast<unsigned long long>((c.protect_pages << 12u) / MiB), per(c.totals[WriteBack]),
	            static_cast<unsigned long long>(c.writeback_bytes / MiB), c.owners.size() - 1, c.runs.size(),
	            c.walk_errors != 0 ? " (some lists changed while read), " : ", ", static_cast<unsigned long long>(c.dropped));
	std::vector<uint32_t> order;
	for (uint32_t i = 0; i < c.owners.size(); i++) {
		const auto& o = c.owners[i];
		if (i == 0 || o.held != 0 || o.events[FaultWindow] + o.events[FaultWrite] + o.events[FaultRead] + o.events[Protect] +
		                                     o.events[WriteBack] != 0)
			order.push_back(i);
	}
	const auto weight = [&](uint32_t i) {
		const auto& o = c.owners[i];
		return o.events[FaultWindow] + o.events[FaultWrite] + o.events[FaultRead] + o.events[Protect] + o.events[WriteBack];
	};
	std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
		return weight(a) != weight(b) ? weight(a) > weight(b) : c.owners[a].held > c.owners[b].held;
	});
	static const char* const Types[] = {"?", "cpu", "gpu"};
	size_t                   shown   = 0;
	for (const auto i: order) {
		const auto& o = c.owners[i];
		const bool  rt = o.name == "RTSharedAllocator" || o.name == "RTPersistAllocator";
		if (shown >= 12 && !rt && i != 0) continue;
		shown++;
		char resident[64] = "";
		if (residency && o.sampled != 0)
			std::snprintf(resident, sizeof(resident), " resident %llu MiB", static_cast<unsigned long long>(o.resident / MiB));
		std::printf("[native-mem]   %-22s #%-3u %-3s held %5llu MiB in %4llu runs%s: per s fw %llu fwr %llu fr %llu prot %llu "
		            "(%llu MiB) wb %llu MiB\n",
		            i == 0 ? "(no engine run)" : o.name.c_str(), o.slot, i == 0 ? "-" : Types[std::min<uint32_t>(o.type, 2)],
		            static_cast<unsigned long long>(o.held / MiB), static_cast<unsigned long long>(o.runs), resident,
		            per(o.events[FaultWindow]), per(o.events[FaultWrite]), per(o.events[FaultRead]), per(o.events[Protect]),
		            static_cast<unsigned long long>((o.protect_pages << 12u) / MiB),
		            static_cast<unsigned long long>(o.writeback_bytes / MiB));
	}
	if (g_level.load(std::memory_order_relaxed) >= 2) {
		std::printf("[native-mem]   GPU-only arenas, kept in video memory instead of downloaded into guest RAM (since "
		            "start): image collector %llu passes (%llu MiB summed over them), linear readback %llu (%llu MiB)\n",
		            static_cast<unsigned long long>(g_kept_images[0].load()),
		            static_cast<unsigned long long>(g_kept_bytes[0].load() / MiB),
		            static_cast<unsigned long long>(g_kept_images[1].load()),
		            static_cast<unsigned long long>(g_kept_bytes[1].load() / MiB));
	}
	std::fflush(stdout);
	for (auto& o: c.owners) {
		std::fill(std::begin(o.events), std::end(o.events), 0);
		o.protect_pages = o.writeback_bytes = 0;
	}
	std::fill(std::begin(c.totals), std::end(c.totals), 0);
	c.protect_pages = c.writeback_bytes = 0;
	c.dropped = c.walk_errors = 0;
}

void CensusThread() {
	Census c;
	c.owners.emplace_back(); // [0]
	c.owners[0].name = "(no engine run)";
	c.tail           = g_head.load(std::memory_order_acquire);
	auto     last    = std::chrono::steady_clock::now();
	uint64_t second  = 0;
	Walk(c);
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		++second;
		Drain(c);
		if (second % 10 != 0) continue;
		const bool residency = second % 30 == 0;
		if (residency) SampleResidency(c);
		const auto   now     = std::chrono::steady_clock::now();
		const double seconds = std::chrono::duration<double>(now - last).count();
		last                 = now;
		Report(c, seconds > 0.5 ? seconds : 10.0, residency);
		Walk(c); // (the runs for the next period)
	}
}

} // namespace

void Record(Event kind, uint64_t vaddr, uint64_t size) noexcept {
	const uint64_t index = g_head.fetch_add(1, std::memory_order_relaxed);
	g_ring[index & (RingSize - 1)].store(Pack(kind, vaddr, size), std::memory_order_relaxed);
}

bool InRtArena(uint64_t vaddr, uint64_t size) noexcept {
	const int count = g_state.arenas.load(std::memory_order_acquire);
	for (int i = 0; i < count; i++) {
		const uint64_t begin = g_state.arena_va[i], end = begin + g_state.arena_size[i];
		if (vaddr >= begin && vaddr < end && size <= end - vaddr) return true;
	}
	return false;
}

bool KeepGpuOnly(uint64_t vaddr, uint64_t size, int why) noexcept {
	if (g_level.load(std::memory_order_relaxed) < 2 || size == 0 || !InRtArena(vaddr, size)) return false;
	const int w = why != 0 ? 1 : 0;
	g_kept_images[w].fetch_add(1, std::memory_order_relaxed);
	g_kept_bytes[w].fetch_add(size, std::memory_order_relaxed);
	return true;
}

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	const char* v     = std::getenv("KYTY_NATIVE_MEM");
	const int   level = v != nullptr && (v[0] == '1' || v[0] == '2') ? v[0] - '0' : 0;
	if (level == 0) return;
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
		g_state.segments.push_back({begin, begin + size});
	}
	std::string version;
	(void)SystemContentParamSfoGetString("APP_VER", &version);
	uint64_t    where  = 0;
	const char* failed = CheckStatic(g_state.base, &where);
	if (failed != nullptr) {
		std::printf("[native-mem] KYTY_NATIVE_MEM=%d NOT applied: eboot+0x%llx: %s (version %s; checked for 01.007.000)\n", level,
		            static_cast<unsigned long long>(where), failed, version.c_str());
		std::fflush(stdout);
		return;
	}
	g_level.store(level, std::memory_order_release);
	std::printf("[native-mem] KYTY_NATIVE_MEM=%d: engine code checked (version %s: render-target arenas, allocator registry, "
	            "page-run lists); %s\n",
	            level, version.c_str(),
	            level >= 2 ? "census, and render-target arenas kept GPU-only once found"
	                       : "census only (nothing in the emulator's behaviour changes)");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

void OnRegisterBuffers() noexcept {
	if (!On() || g_state.probed.exchange(true)) return;
	const uint64_t base = g_state.base;
	bool           ok   = true;
	const uint64_t mgr  = ReadU64(base + RtManagerGlobal, &ok);
	// The arena vtable's relocated slots: the bump allocator and the empty Free.
	const uint64_t alloc_slot = ReadU64(base + ArenaVtable + 0xa8, &ok);
	const uint64_t free_slot  = ReadU64(base + ArenaVtable + 0xb0, &ok);
	const char*    failed     = nullptr;
	if (!ok || mgr == 0) failed = "no RenderTargetManager";
	else if (alloc_slot != base + ArenaAlloc || free_slot != base + ArenaFree) failed = "arena vtable slots differ";
	struct Arena {
		uint64_t    member, name;
		uint32_t    flag;
		uint64_t    expected;
		const char* label;
	};
	const Arena arenas[2] = {{0x88, SharedName, 1, ExpectedShared, "RTSharedAllocator"},
	                         {0xa8, PersistName, 0, ExpectedPersist, "RTPersistAllocator"}};
	std::array<uint64_t, 2> va {}, size {};
	for (int i = 0; i < 2 && failed == nullptr; i++) {
		const uint64_t object = ReadU64(mgr + arenas[i].member, &ok, true);
		uint8_t        o[0xc0] {};
		if (!ok || object == 0 || !ReadGuest(object, o, sizeof(o), true)) {
			failed = "arena object unreadable";
			break;
		}
		uint64_t vtable = 0, name = 0;
		uint32_t type = 0, flag = 0;
		std::memcpy(&vtable, o, 8);
		std::memcpy(&name, o + 0x10, 8);
		std::memcpy(&type, o + 0x18, 4);
		std::memcpy(&va[i], o + 0x88, 8);
		std::memcpy(&size[i], o + 0x90, 8);
		std::memcpy(&flag, o + 0x98, 4);
		if (vtable != base + ArenaVtable) failed = "arena vtable differs";
		else if (name != base + arenas[i].name) failed = "arena name differs";
		else if (type != 2) failed = "arena EMemoryType is not GPUMemory";
		else if (flag != arenas[i].flag) failed = "arena shared flag differs";
		else if (va[i] == 0 || size[i] == 0 || ((va[i] | size[i]) & 0xffffu) != 0 || size[i] > (uint64_t {4} << 30u))
			failed = "arena range invalid (InitSharedMemory's RequestPages failed?)";
		else if (!Libs::LibKernel::Memory::IsFullyMapped(va[i], size[i])) failed = "arena range not mapped";
	}
	if (failed == nullptr && va[0] < va[1] + size[1] && va[1] < va[0] + size[0]) failed = "arenas overlap";
	if (failed != nullptr) {
		std::printf("[native-mem] render-target arenas NOT found: %s (census of the other allocators only)\n", failed);
	} else {
		for (int i = 0; i < 2; i++) {
			g_state.arena_va[i]   = va[i];
			g_state.arena_size[i] = size[i];
			std::printf("[native-mem] %s: 0x%010llx + %llu MiB (mode 4 gives %llu MiB): GPU-only on a native port%s\n",
			            arenas[i].label, static_cast<unsigned long long>(va[i]), static_cast<unsigned long long>(size[i] / MiB),
			            static_cast<unsigned long long>(arenas[i].expected / MiB),
			            g_level.load() >= 2 ? ", kept in video memory (never downloaded into guest RAM)" : "");
		}
		g_state.arenas.store(2, std::memory_order_release);
	}
	std::fflush(stdout);
	std::thread(CensusThread).detach();
}

} // namespace Loader::DemonsSoulsNativeMem
